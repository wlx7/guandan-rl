//! Guandan (掼蛋) rules engine with Python bindings.

pub mod bot;
pub mod cards;
pub mod endgame;
pub mod features;
pub mod game;
pub mod moves;
pub mod search;
pub mod venv;

#[cfg(feature = "python")]
mod py;

/// Version of the whole project (the Python package and the Android app use the same number).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
