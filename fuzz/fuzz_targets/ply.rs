#![no_main]

use libfuzzer_sys::fuzz_target;

// The parser must reject malformed input with an error, never panic.
fuzz_target!(|data: &[u8]| {
    let _ = tpt_percept_io::read_ply_from(std::io::Cursor::new(data));
});
