use std::borrow::Cow;

use bevy::{
  DefaultPlugins,
  app::{App, Plugin, Startup, Update},
  asset::{AssetServer, Assets, Handle, RenderAssetUsages},
  camera::Camera2d,
  color::Color,
  ecs::{
    resource::Resource,
    schedule::IntoScheduleConfigs,
    system::{Commands, Res, ResMut, Single},
    world::World,
  },
  image::Image,
  math::{Vec2, Vec3},
  prelude::{PluginGroup, default},
  render::{
    Render, RenderApp, RenderStartup, RenderSystems,
    camera::ClearColor,
    extract_resource::{ExtractResource, ExtractResourcePlugin},
    render_asset::RenderAssets,
    render_graph::{self, RenderGraph, RenderLabel},
    render_resource::{
      BindGroup, BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries,
      CachedComputePipelineId, CachedPipelineState, ComputePassDescriptor,
      ComputePipelineDescriptor, PipelineCache, ShaderStages, StorageTextureAccess, TextureFormat,
      TextureUsages, binding_types::texture_storage_2d,
    },
    renderer::{RenderContext, RenderDevice},
    texture::GpuImage,
  },
  shader::PipelineCacheError,
  sprite::Sprite,
  transform::components::Transform,
  window::{Window, WindowPlugin},
};

const SHADER_ASSET_PATH: &str = "mandelbrot.wgsl";

const DISPLAY_FACTOR: u32 = 4;
const SIZE: (u32, u32) = (1280 / DISPLAY_FACTOR, 720 / DISPLAY_FACTOR);
const WORKGROUP_SIZE: u32 = 8;

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
    .add_plugins(MandelbrotComputePlugin)
    .add_systems(Startup, setup)
    .add_systems(Update, switch_textures)
    .run();
}

fn setup(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
  let mut image = Image::new_target_texture(SIZE.0, SIZE.1, TextureFormat::Rgba32Float);
  image.asset_usage = RenderAssetUsages::RENDER_WORLD;
  image.texture_descriptor.usage =
    TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;
  let image0 = images.add(image.clone());
  let image1 = images.add(image);

  commands.spawn((
    Sprite {
      image: image0.clone(),
      custom_size: Some(Vec2::new(SIZE.0 as f32, SIZE.1 as f32)),
      ..default()
    },
    Transform::from_scale(Vec3::splat(DISPLAY_FACTOR as f32)),
  ));
  commands.spawn(Camera2d);

  commands.insert_resource(MandelbrotImages { texture_a: image0, texture_b: image1 });
}

fn switch_textures(images: Res<MandelbrotImages>, mut sprite: Single<&mut Sprite>) {
  if sprite.image == images.texture_a {
    sprite.image = images.texture_b.clone();
  } else {
    sprite.image = images.texture_a.clone();
  }
}

#[derive(Resource, Clone, ExtractResource)]
struct MandelbrotImages {
  texture_a: Handle<Image>,
  texture_b: Handle<Image>,
}

#[derive(Resource)]
struct MandelbrotImageBindGroups([BindGroup; 2]);

fn prepare_bind_group(
  mut commands: Commands,
  pipeline: Res<MandelbrotPipeline>,
  gpu_images: Res<RenderAssets<GpuImage>>,
  game_of_life_images: Res<MandelbrotImages>,
  render_device: Res<RenderDevice>,
) {
  let view_a = gpu_images.get(&game_of_life_images.texture_a).unwrap();
  let view_b = gpu_images.get(&game_of_life_images.texture_b).unwrap();
  let bind_group_0 = render_device.create_bind_group(
    None,
    &pipeline.texture_bind_group_layout,
    &BindGroupEntries::sequential((&view_a.texture_view, &view_b.texture_view)),
  );
  let bind_group_1 = render_device.create_bind_group(
    None,
    &pipeline.texture_bind_group_layout,
    &BindGroupEntries::sequential((&view_b.texture_view, &view_a.texture_view)),
  );
  commands.insert_resource(MandelbrotImageBindGroups([bind_group_0, bind_group_1]));
}

struct MandelbrotComputePlugin;

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct MandelbrotLabel;

impl Plugin for MandelbrotComputePlugin {
  fn build(&self, app: &mut App) {
    app.add_plugins(ExtractResourcePlugin::<MandelbrotImages>::default());
    let render_app = app.sub_app_mut(RenderApp);
    render_app
      .add_systems(RenderStartup, init_mandelbrot_pipeline)
      .add_systems(
        Render,
        prepare_bind_group.in_set(RenderSystems::PrepareBindGroups),
      );

    let mut render_graph = render_app.world_mut().resource_mut::<RenderGraph>();
    render_graph.add_node(MandelbrotLabel, MandelbrotNode::default());
    render_graph.add_node_edge(MandelbrotLabel, bevy::render::graph::CameraDriverLabel);
  }
}

#[derive(Resource)]
struct MandelbrotPipeline {
  texture_bind_group_layout: BindGroupLayout,
  checker_board_pipeline: CachedComputePipelineId,
}

fn init_mandelbrot_pipeline(
  mut commands: Commands,
  render_device: Res<RenderDevice>,
  asset_server: Res<AssetServer>,
  pipeline_cache: Res<PipelineCache>,
) {
  let texture_bind_group_layout = render_device.create_bind_group_layout(
    "Mandelbrot",
    &BindGroupLayoutEntries::sequential(
      ShaderStages::COMPUTE,
      (texture_storage_2d(
        TextureFormat::Rgba32Float,
        StorageTextureAccess::WriteOnly,
      ),),
    ),
  );

  let shader = asset_server.load(SHADER_ASSET_PATH);
  let checker_board_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
    layout: vec![texture_bind_group_layout.clone()],
    shader: shader,
    entry_point: Some(Cow::from("checker_board")),
    ..default()
  });

  commands.insert_resource(MandelbrotPipeline {
    texture_bind_group_layout,
    checker_board_pipeline,
  });
}

enum MandelbrotState {
  Loading,
  Update(usize),
}

struct MandelbrotNode {
  state: MandelbrotState,
}

impl Default for MandelbrotNode {
  fn default() -> Self {
    Self { state: MandelbrotState::Loading }
  }
}

impl render_graph::Node for MandelbrotNode {
  fn update(&mut self, world: &mut World) {
    let pipeline = world.resource::<MandelbrotPipeline>();
    let pipeline_cache = world.resource::<PipelineCache>();

    // if the corresponding pipeline has loaded, transition to the next stage
    match self.state {
      MandelbrotState::Loading => {
        match pipeline_cache.get_compute_pipeline_state(pipeline.checker_board_pipeline) {
          CachedPipelineState::Ok(_) => {
            self.state = MandelbrotState::Update(0);
          }
          // If the shader hasn't loaded yet, just wait.
          CachedPipelineState::Err(PipelineCacheError::ShaderNotLoaded(_)) => {}
          CachedPipelineState::Err(err) => {
            panic!("Initializing assets/{SHADER_ASSET_PATH}:\n{err}")
          }
          _ => {}
        }
      }
      MandelbrotState::Update(0) => {
        self.state = MandelbrotState::Update(1);
      }
      MandelbrotState::Update(1) => {
        self.state = MandelbrotState::Update(0);
      }
      MandelbrotState::Update(_) => unreachable!(),
    }
  }

  fn run(
    &self,
    _graph: &mut render_graph::RenderGraphContext,
    render_context: &mut RenderContext,
    world: &World,
  ) -> Result<(), render_graph::NodeRunError> {
    let bind_groups = &world.resource::<MandelbrotImageBindGroups>().0;
    let pipeline_cache = world.resource::<PipelineCache>();
    let pipeline = world.resource::<MandelbrotPipeline>();

    let mut pass = render_context
      .command_encoder()
      .begin_compute_pass(&ComputePassDescriptor::default());

    match self.state {
      MandelbrotState::Loading => {}
      MandelbrotState::Update(index) => {
        let checker_board_pipeline = pipeline_cache
          .get_compute_pipeline(pipeline.checker_board_pipeline)
          .unwrap();
        pass.set_bind_group(0, &bind_groups[index], &[]);
        pass.set_pipeline(checker_board_pipeline);
        pass.dispatch_workgroups(SIZE.0 / WORKGROUP_SIZE, SIZE.1 / WORKGROUP_SIZE, 1);
      }
    }

    Ok(())
  }
}
