use lang_c::driver::{Config, parse};
use std::env;
use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

mod pass;
mod vlir;

fn print_usage() {
    eprintln!("usage: compiler <input.c> [output.json] [options]");
    eprintln!("  Compiles the first `kernel()` in the translation unit to VLIW JSON.");
    eprintln!(
        "  Other top-level functions may appear as `__builtin_spawn(..., name, ...)` targets."
    );
    eprintln!("  If output.json is omitted, writes to output/compiled_<unix_time>.json");
    eprintln!(
        "  JSON: {{ instructions: [ bundles ], debug_info: {{ scratch_map: {{ addr: [name, len] }} }} }}."
    );
    eprintln!("options:");
    eprintln!(
        "  --no-schedule   Disable advanced VLIW instruction scheduling (one IR op per bundle)."
    );
    eprintln!("                  Default: scheduling enabled.");
    eprintln!(
        "  --trace-scratch Record scratch allocator alloc/use/free in debug_info.scratch_lifetime."
    );
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1).collect::<Vec<_>>();
    if args.is_empty() || args[0] == "-h" || args[0] == "--help" {
        print_usage();
        if args.is_empty() {
            std::process::exit(2);
        }
        return Ok(());
    }

    let mut advanced_scheduling = true;
    let mut trace_scratch = false;
    args.retain(|a| {
        if a == "--no-schedule" {
            advanced_scheduling = false;
            return false;
        }
        if a == "--trace-scratch" {
            trace_scratch = true;
            return false;
        }
        true
    });

    let input_path = if !args.is_empty() {
        args.remove(0)
    } else {
        "test/vector_lanes.c".to_string()
    };
    let out_path = if !args.is_empty() {
        args.remove(0)
    } else {
        let out_dir = "output";
        fs::create_dir_all(out_dir)?;
        let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
        format!("{out_dir}/compiled_{now}.json")
    };

    let config = Config::default();
    let mut ast = parse(&config, &input_path)?;
    pass::spawn_expand::expand_spawn_in_kernel(&mut ast.unit)
        .map_err(|e| format!("spawn expansion error: {e}"))?;
    // let s = &mut String::new();
    // Printer::new(s).visit_translation_unit(&ast.unit);
    // println!("{}", s);
    let program = vlir::lowering::lower_translation_unit(&ast.unit)
        .map_err(|e| format!("AST lowering error: {e:?}"))?;

    let machine_programs: Result<Vec<_>, _> = program
        .functions
        .iter()
        .map(|f| {
            if trace_scratch {
                f.lower_to_machine_traced(advanced_scheduling)
                    .map(|(p, d, t)| (p, d, Some(t)))
            } else {
                f.lower_to_machine(advanced_scheduling)
                    .map(|(p, d)| (p, d, None))
            }
        })
        .collect();
    let machine_programs =
        machine_programs.map_err(|e| format!("Machine lowering error: {e:?}"))?;

    if machine_programs.len() != 1 {
        eprintln!(
            "warning: expected exactly one `kernel` function, got {}",
            machine_programs.len()
        );
    }

    let Some((prog, scratch_debug, scratch_trace)) = machine_programs.first() else {
        return Err("no machine program produced (missing kernel()?)".into());
    };

    let json = vlir::machine::InstructionBundle::program_to_json_with_debug_and_trace(
        prog,
        scratch_debug,
        scratch_trace.as_ref(),
    );
    if let Some(parent) = Path::new(&out_path).parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&out_path, json)?;
    println!("wrote {}", out_path);

    Ok(())
}
