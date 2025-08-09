use std::{
  borrow::Cow,
  sync::atomic::{AtomicBool, Ordering},
};

use bevy::{
  DefaultPlugins,
  app::{App, Plugin, Startup},
  asset::{AssetMetaCheck, AssetMode, AssetPlugin, AssetServer, Assets, Handle, RenderAssetUsages},
  camera::Camera2d,
  color::Color,
  ecs::{
    resource::Resource,
    schedule::IntoScheduleConfigs,
    system::{Commands, Res, ResMut},
    world::World,
  },
  image::Image,
  log::info,
  math::{Vec2, Vec3},
  prelude::{PluginGroup, default},
  render::{
    Render, RenderApp, RenderStartup, RenderSystems,
    camera::ClearColor,
    extract_resource::{ExtractResource, ExtractResourcePlugin},
    render_asset::RenderAssets,
    render_graph::{self, RenderGraph, RenderLabel},
    render_resource::{
      BindGroup, BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries, Buffer,
      BufferDescriptor, BufferInitDescriptor, BufferUsages, CachedComputePipelineId,
      CachedPipelineState, ComputePassDescriptor, ComputePipelineDescriptor, MapMode,
      PipelineCache, PollType, ShaderStages, ShaderType, StorageTextureAccess, TextureFormat,
      TextureUsages,
      binding_types::{texture_storage_2d, uniform_buffer},
    },
    renderer::{RenderContext, RenderDevice},
    texture::GpuImage,
    view::Msaa,
  },
  shader::PipelineCacheError,
  sprite::Sprite,
  transform::components::Transform,
  window::{Window, WindowPlugin},
};
use bytemuck::{Pod, Zeroable, bytes_of};

const SHADER_ASSET_PATH: &str = "mandelbrot.wgsl";

const DISPLAY_FACTOR: u32 = 1;
const SIZE: (u32, u32) = (1280 / DISPLAY_FACTOR, 720 / DISPLAY_FACTOR);
const WORKGROUP_SIZE: u32 = 8;

#[derive(Resource, Clone, Copy, Pod, Zeroable, ShaderType)]
#[repr(C)]
struct Uniforms {
  time: u32,
}

fn main() {
  App::new()
    .insert_resource(ClearColor(Color::BLACK))
    .add_plugins(
      DefaultPlugins
        .set(WindowPlugin {
          primary_window: Some(Window {
            resolution: (1280., 720.).into(),
            ..default()
          }),
          ..default()
        })
        .set(AssetPlugin {
          mode: AssetMode::Unprocessed,
          meta_check: AssetMetaCheck::Never,
          ..default()
        }),
    )
    .add_plugins(MandelbrotComputePlugin)
    .add_systems(Startup, setup)
    .run();
}

fn setup(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
  let mut image = Image::new_target_texture(SIZE.0, SIZE.1, TextureFormat::Rgba32Float);
  image.asset_usage = RenderAssetUsages::RENDER_WORLD;
  image.texture_descriptor.usage =
    TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;
  let image = images.add(image);

  commands.spawn((
    Sprite {
      image: image.clone(),
      custom_size: Some(Vec2::new(SIZE.0 as f32, SIZE.1 as f32)),
      ..default()
    },
    Transform::from_scale(Vec3::splat(DISPLAY_FACTOR as f32)),
  ));
  commands.spawn((Msaa::Off, Camera2d));

  commands.insert_resource(MandelbrotImages { texture: image });
}

#[derive(Resource, Clone, ExtractResource)]
struct MandelbrotImages {
  texture: Handle<Image>,
}

#[derive(Resource)]
struct MandelbrotImageBindGroups(BindGroup);

fn prepare_bind_group(
  mut commands: Commands,
  pipeline: Res<MandelbrotPipeline>,
  gpu_images: Res<RenderAssets<GpuImage>>,
  game_of_life_images: Res<MandelbrotImages>,
  render_device: Res<RenderDevice>,
  mut uniform_data: ResMut<Uniforms>,
) {
  uniform_data.time += 1;

  let view = gpu_images.get(&game_of_life_images.texture).unwrap();
  let bind_group_0 = render_device.create_bind_group(
    None,
    &pipeline.texture_bind_group_layout,
    &BindGroupEntries::sequential((
      &view.texture_view,
      pipeline.uniform_buffer.as_entire_buffer_binding(),
    )),
  );
  commands.insert_resource(MandelbrotImageBindGroups(bind_group_0));
}

fn update_uniforms(
  pipeline: Res<MandelbrotPipeline>,
  uniform_data: Res<Uniforms>,
  render_device: Res<RenderDevice>,
) {
  static MAPPED: AtomicBool = AtomicBool::new(false);

  let uniform_data = *uniform_data;
  let buffer = pipeline.mapped_uniform_buffer.clone();
  if MAPPED.swap(true, Ordering::SeqCst) {
    return;
  }

  info!("Tryna read map!");
  // Maps the buffer so it can be read on the cpu
  pipeline
    .mapped_uniform_buffer
    .slice(..)
    .map_async(MapMode::Write, move |r| match r {
      // This will execute once the gpu is ready, so after the call to poll()
      Ok(_) => {
        info!("Read map!");
        buffer
          .slice(..)
          .get_mapped_range_mut()
          .copy_from_slice(bytes_of(&uniform_data));

        buffer.unmap();
        info!("Unmapped buffer");
        MAPPED.store(false, Ordering::SeqCst);
      }
      Err(err) => panic!("Failed to map buffer {err}"),
    });

  render_device
    .poll(PollType::Wait)
    .expect("Failed to wait for render device");
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
        (
          prepare_bind_group.in_set(RenderSystems::PrepareBindGroups),
          update_uniforms.after(RenderSystems::Render),
        ),
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
  uniform_buffer: Buffer,
  mapped_uniform_buffer: Buffer,
}

fn init_mandelbrot_pipeline(
  mut commands: Commands,
  render_device: Res<RenderDevice>,
  asset_server: Res<AssetServer>,
  pipeline_cache: Res<PipelineCache>,
) {
  let uniforms = Uniforms { time: 0 };
  let buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
    label: Some("Uniforms"),
    contents: bytes_of(&uniforms),
    usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
  });
  let mapped_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
    label: Some("Mapped uniforms"),
    contents: bytes_of(&uniforms),
    usage: BufferUsages::MAP_WRITE | BufferUsages::COPY_SRC,
  });
  commands.insert_resource(uniforms);

  let texture_bind_group_layout = render_device.create_bind_group_layout(
    "Mandelbrot",
    &BindGroupLayoutEntries::sequential(
      ShaderStages::COMPUTE,
      (
        texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly),
        uniform_buffer::<Uniforms>(false),
      ),
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
    uniform_buffer: buffer,
    mapped_uniform_buffer: mapped_buffer,
  });
}

enum MandelbrotState {
  Loading,
  Update,
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
            self.state = MandelbrotState::Update;
          }
          // If the shader hasn't loaded yet, just wait.
          CachedPipelineState::Err(PipelineCacheError::ShaderNotLoaded(_)) => {}
          CachedPipelineState::Err(err) => {
            panic!("Initializing assets/{SHADER_ASSET_PATH}:\n{err}")
          }
          _ => {}
        }
      }
      MandelbrotState::Update => {}
    }
  }

  fn run(
    &self,
    _graph: &mut render_graph::RenderGraphContext,
    render_context: &mut RenderContext,
    world: &World,
  ) -> Result<(), render_graph::NodeRunError> {
    let bind_group = &world.resource::<MandelbrotImageBindGroups>().0;
    let pipeline_cache = world.resource::<PipelineCache>();
    let pipeline = world.resource::<MandelbrotPipeline>();

    render_context.command_encoder().copy_buffer_to_buffer(
      &pipeline.mapped_uniform_buffer,
      0,
      &pipeline.uniform_buffer,
      0,
      size_of::<Uniforms>() as u64,
    );

    let mut pass = render_context
      .command_encoder()
      .begin_compute_pass(&ComputePassDescriptor::default());

    match self.state {
      MandelbrotState::Loading => {}
      MandelbrotState::Update => {
        let checker_board_pipeline = pipeline_cache
          .get_compute_pipeline(pipeline.checker_board_pipeline)
          .unwrap();
        pass.set_bind_group(0, bind_group, &[]);
        pass.set_pipeline(checker_board_pipeline);
        pass.dispatch_workgroups(SIZE.0 / WORKGROUP_SIZE, SIZE.1 / WORKGROUP_SIZE, 1);
      }
    }

    Ok(())
  }
}
