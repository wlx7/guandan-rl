//! Guandan (掼蛋) rules engine with Python bindings.

pub mod bot;
pub mod cards;
pub mod features;
pub mod game;
pub mod moves;
pub mod venv;

#[cfg(feature = "python")]
mod py;
