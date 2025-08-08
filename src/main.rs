use bevy::{
  DefaultPlugins,
  app::App,
  color::Color,
  prelude::{PluginGroup, default},
  render::camera::ClearColor,
  window::{Window, WindowPlugin},
};

fn main() {
  App::new()
    .insert_resource(ClearColor(Color::BLACK))
    .add_plugins(DefaultPlugins.set(WindowPlugin {
      primary_window: Some(Window {
        resolution: (1280., 720.).into(),
        ..default()
      }),
      ..default()
    }))
    .run();
}
