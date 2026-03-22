use lang_c::driver::{Config, parse};
use std::env;
use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

mod pass;
mod vlir;

fn print_usage() {
    eprintln!("usage: compiler <input.c> [output.json]");
    eprintln!("  Compiles the first `kernel()` in the translation unit to VLIW JSON.");
    eprintln!("  If output.json is omitted, writes to output/compiled_<unix_time>.json");
    eprintln!("  JSON: {{ instructions: [ bundles ], debug_info: {{ scratch_map: {{ addr: [name, len] }} }} }}.");
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

    let input_path = args.remove(0);
    let out_path = if args.is_empty() {
        let out_dir = "output";
        fs::create_dir_all(out_dir)?;
        let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
        format!("{out_dir}/compiled_{now}.json")
    } else {
        args.remove(0)
    };

    let config = Config::default();
    let ast = parse(&config, &input_path)?;
    let program = vlir::lowering::lower_translation_unit(&ast.unit)
        .map_err(|e| format!("AST lowering error: {e:?}"))?;

    let machine_programs = program
        .lower_to_machine()
        .map_err(|e| format!("Machine lowering error: {e:?}"))?;

    if machine_programs.len() != 1 {
        eprintln!(
            "warning: expected exactly one `kernel` function, got {}",
            machine_programs.len()
        );
    }

    let Some((prog, scratch_debug)) = machine_programs.first() else {
        return Err("no machine program produced (missing kernel()?)".into());
    };

    let json =
        vlir::machine::InstructionBundle::program_to_json_with_debug(prog, scratch_debug);
    if let Some(parent) = Path::new(&out_path).parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&out_path, json)?;
    println!("wrote {}", out_path);

    Ok(())
}
