use bytemuck;
use color_eyre::eyre;
use lyon::lyon_tessellation::{BuffersBuilder, FillVertex};
use lyon::tessellation::{FillOptions, FillTessellator, VertexBuffers};
use wgpu::util::DeviceExt;
// use lyon::geom::euclid::UnknownUnit;
use lyon::math::point;

// Vertices + Indices
type TessellatedOutput = (Vec<Vertex>, Vec<u16>);

#[derive(Debug, Copy, Clone)]
struct Color {
    r: f32,
    g: f32,
    b: f32,
}

impl Color {
    fn new(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b }
    }

    fn as_array(self) -> [f32; 3] {
        [self.r, self.g, self.b]
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
}

impl Vertex {
    /// Return the `wgpu::VertexBufferLayout` that can be used to describe this Vertex.
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            // 'array_stride' defines how wide a Vertex is. Right now, this will probably be
            // 24 bytes (8 bytes * 3 components)
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            // 'step_mode' tells the pipeline whether each elemenet in the array
            // of this buffer represents per-vertex data of per-instance data
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                // Vertex::position
                wgpu::VertexAttribute {
                    offset: 0,
                    // This tells the shader where to store this attribute.
                    // For example `@location(0) x: vec3<f32> would correspond to the
                    // `position` field of the Vertex struct, while `@location(1) x: vec<f32`
                    // would correspond to the color field
                    shader_location: 0,
                    // Float32x3 corresponds to a vec3<f32> in shader code
                    format: wgpu::VertexFormat::Float32x3,
                },
                // Vertex::color
                wgpu::VertexAttribute {
                    // This attribute comes (in memory) after the size of 'position',
                    // which is [f32; 3] currently
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

#[derive(Debug, Clone)]
enum CustomSketchEvent {
    /// A user request to re-render the frame
    Rerender(String),
}

pub struct GPUState {
    pub surface: wgpu::Surface,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface_config: wgpu::SurfaceConfiguration,
    pub size: winit::dpi::PhysicalSize<u32>,
    pub window: winit::window::Window,
    pub shader: wgpu::ShaderModule,
    pub render_pipeline: wgpu::RenderPipeline,
    vertices: Vec<Vertex>,
    indices: Vec<u16>,
    fill_color: Color,
}

impl GPUState {
    async fn new(window: winit::window::Window) -> Self {
        let size = window.inner_size();

        // This is a handle to our GPU
        // All backends means Vulkan | Metal | DX12 | Browser WebGPU
        let gpu_instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            dx12_shader_compiler: Default::default(),
        });

        // The surface needs to live as long as the window that created it.
        // Since this struct owns the window, this should be safe in practice
        // (but not in theory)
        let surface = unsafe { gpu_instance.create_surface(&window) }.unwrap();

        let adapter_options = wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        };

        let gpu_adapter = gpu_instance
            .request_adapter(&adapter_options)
            .await
            .unwrap();

        let device_desc = wgpu::DeviceDescriptor {
            features: wgpu::Features::empty(),
            limits: wgpu::Limits::default(),
            label: None,
        };

        let trace_path = None;
        let (device, queue) = gpu_adapter
            .request_device(&device_desc, trace_path)
            .await
            .unwrap();

        // Pick a texture format supported by the current adapter
        // For more context on Bgra formats, see https://github.com/gfx-rs/wgpu-rs/issues/123
        let surface_caps = surface.get_capabilities(&gpu_adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            // The usage field describes how SurfaceTextures will be used.
            // RENDER_ATTACHMENT specifies that the textures will be used to write
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            // width and height are the width and the height in pixels of a SurfaceTexture.
            // This should usually be the width and the height of the window.
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        let shader_source = include_str!("shader.wgsl");
        let shader_desc = wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        };

        let shader = device.create_shader_module(shader_desc);

        // Render Pipeline
        let pipeline_layout_desc = wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        };

        let render_pipeline_layout = device.create_pipeline_layout(&pipeline_layout_desc);

        let target_states = vec![Some(wgpu::ColorTargetState {
            format: surface_format,
            blend: Some(wgpu::BlendState::REPLACE),
            write_mask: wgpu::ColorWrites::ALL,
        })];

        let topology = wgpu::PrimitiveTopology::TriangleStrip;

        let render_pipeline_desc = wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                // The 'buffers' field tells wgpu what type of vertices
                // we want to pass to the vertex shader.
                buffers: &[Vertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                // The 'targets' field tells wgpu what color outputs it should
                // set up. Currently, we only need one for the surface. We
                // use the surface's format so that copying to it is easy,
                // and we specify that the blending should just replace old
                // pixel data with new data. We also tell wgpu to write to
                // all colors: red, blue, green, and alpha.
                targets: &target_states,
            }),
            // The 'primitive' field  describes how to interpret the vertices
            // we're gonna send when converting them into triangles
            primitive: wgpu::PrimitiveState {
                topology,
                strip_index_format: None,
                // FrontFace::Ccw means that a triangle is facing forward
                // if the vertices are arranged in a counter-clockwise direction.
                // Triangles that are not facing forward will be culled.
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        };

        let render_pipeline = device.create_render_pipeline(&render_pipeline_desc);

        let vertices = vec![
            // Top
            Vertex {
                position: [0.0, 0.5, 0.0],
                color: [1.0, 0.0, 0.0],
            },
            // Bottom left
            Vertex {
                position: [-0.5, -0.5, 0.0],
                color: [0.0, 1.0, 0.0],
            },
            // Bottom right
            Vertex {
                position: [0.5, -0.5, 0.0],
                color: [0.0, 0.0, 1.0],
            },
        ];

        let indices = vec![0, 1, 2];
        let fill_color = Color::new(1.0, 0.0, 1.0);

        Self {
            window,
            surface,
            device,
            queue,
            surface_config: config,
            size,
            shader,
            render_pipeline,
            vertices,
            indices,
            fill_color,
        }
    }

    pub fn window(&self) -> &winit::window::Window {
        &self.window
    }

    fn update_vertices(&mut self, vertices: Vec<Vertex>) {
        self.vertices = vertices;
    }

    fn update_indices(&mut self, indices: Vec<u16>) {
        self.indices = indices;
    }

    fn create_render_pipeline(&mut self) {
        let pipeline_layout_desc = wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        };

        let render_pipeline_layout = self.device.create_pipeline_layout(&pipeline_layout_desc);

        let target_states = vec![Some(wgpu::ColorTargetState {
            format: self.surface_config.format.clone(),
            blend: Some(wgpu::BlendState::REPLACE),
            write_mask: wgpu::ColorWrites::ALL,
        })];

        // Decide the topology
        // let topology = wgpu::PrimitiveTopology::TriangleList;
        let topology = wgpu::PrimitiveTopology::TriangleStrip;

        let render_pipeline_desc = wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &self.shader,
                entry_point: "vs_main",
                // The 'buffers' field tells wgpu what type of vertices
                // we want to pass to the vertex shader.
                buffers: &[Vertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &self.shader,
                entry_point: "fs_main",
                // The 'targets' field tells wgpu what color outputs it should
                // set up. Currently, we only need one for the surface. We
                // use the surface's format so that copying to it is easy,
                // and we specify that the blending should just replace old
                // pixel data with new data. We also tell wgpu to write to
                // all colors: red, blue, green, and alpha.
                targets: &target_states,
            }),
            // The 'primitive' field  describes how to interpret the vertices
            // we're gonna send when converting them into triangles
            primitive: wgpu::PrimitiveState {
                topology,
                strip_index_format: None,
                // FrontFace::Ccw means that a triangle is facing forward
                // if the vertices are arranged in a counter-clockwise direction.
                // Triangles that are not facing forward will be culled.
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        };

        self.render_pipeline = self.device.create_render_pipeline(&render_pipeline_desc);
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let vertices = self.vertices.as_slice();
        let indices = self.indices.as_slice();

        // Create the Vertex Buffer to send to the Gpu
        let vertex_buffer_desc = wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX,
        };

        let index_buffer_desc = wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(indices),
            usage: wgpu::BufferUsages::INDEX,
        };

        let vertex_buffer = self.device.create_buffer_init(&vertex_buffer_desc);
        let index_buffer = self.device.create_buffer_init(&index_buffer_desc);
        let num_vertices = vertices.len() as u32;
        let num_indices = indices.len() as u32;

        // Find the texture where we should draw
        let surface_texture = self.surface.get_current_texture()?;
        let view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let command_encoder_desc = wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        };
        let mut encoder = self.device.create_command_encoder(&command_encoder_desc);

        // The RenderPass has all the methods for drawing,
        // and will be sent to the Queue which can then be submitted
        // to the gpu
        let render_pass_desc = wgpu::RenderPassDescriptor {
            label: Some("Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                // The view represents the texture that we are gonna draw this pass onto
                view: &view,
                // The resolve_target is the texture that will receive the resolved output.
                // This will be the same as view unless multisampling is enabled.
                // We don't need to specify this, so we leave it as None.
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 1.0,
                        g: 1.0,
                        b: 1.0,
                        a: 1.0,
                    }),
                    // The store field tells wgpu whether we want to store the rendered
                    // results to the Texture behind our TextureView (in this case it's the SurfaceTexture).
                    // We use true as we do want to store our render results.
                    store: true,
                },
            })],
            depth_stencil_attachment: None,
        };

        let mut render_pass = encoder.begin_render_pass(&render_pass_desc);

        // Begin to actually draw stuff
        render_pass.set_pipeline(&self.render_pipeline);

        render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
        // NOTE: you can only have one index buffer set at a time
        render_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        render_pass.draw_indexed(0..num_indices, 0, 0..1);

        // Draw something with N vertices, and 1 instance
        render_pass.draw(0..num_vertices, 0..1);

        // begin_render_pass() borrows encoder mutably (aka &mut self).
        // We can't call encoder.finish() until we release that mutable borrow.
        // The block tells rust to drop any variables within it when the code
        // leaves that scope thus releasing the mutable borrow on encoder and
        // allowing us to finish() it
        drop(render_pass);

        // Submit accepts anything that implements IntoIterator
        self.queue.submit(std::iter::once(encoder.finish()));

        surface_texture.present();

        Ok(())
    }
}

fn generate_circle(center_xy: (f32, f32), radius: f32, color: &Color) -> TessellatedOutput {
    let mut geometry: VertexBuffers<Vertex, u16> = VertexBuffers::new();

    let mut tessellator = FillTessellator::new();
    let options = FillOptions::default().with_tolerance(0.001);

    tessellator
        .tessellate_circle(
            point(center_xy.0, center_xy.1),
            radius,
            &options,
            &mut BuffersBuilder::new(&mut geometry, |vertex: FillVertex| {
                let p = vertex.position().to_3d();
                let c = color.clone().as_array();
                Vertex {
                    position: p.into(),
                    color: c,
                }
            }),
        )
        .unwrap();

    eprintln!(
        "-- Circle: {} vertices {} indices",
        geometry.vertices.len(),
        geometry.indices.len()
    );

    (geometry.vertices.to_vec(), geometry.indices.to_vec())
}

#[tokio::main]
async fn main() -> eyre::Result<()> {
    // Install the eyre error handlers
    color_eyre::install()?;

    let event_loop =
        winit::event_loop::EventLoopBuilder::<CustomSketchEvent>::with_user_event().build();

    let initial_position = winit::dpi::PhysicalPosition { x: 10, y: 100 };

    let window = winit::window::WindowBuilder::new()
        .with_title("Circle Test")
        .with_position(initial_position)
        .with_resizable(false)
        .build(&event_loop)
        .unwrap();

    let proxy = event_loop.create_proxy();

    let mut gpu_state = GPUState::new(window).await;

    proxy
        .send_event(CustomSketchEvent::Rerender("test".to_string()))
        .unwrap();

    // Event Loop for the Window
    event_loop.run(move |event, _, control_flow| {
        // Event management
        match event {
            // Custom app events
            winit::event::Event::UserEvent(CustomSketchEvent::Rerender(_)) => {
                let radius = 1.0;
                let (vertices, indices) =
                    generate_circle((0.0, 0.0), radius, &gpu_state.fill_color);

                if !vertices.is_empty() && !indices.is_empty() {
                    gpu_state.update_vertices(vertices);
                    gpu_state.update_indices(indices);
                    gpu_state.create_render_pipeline();
                    gpu_state.window().request_redraw();
                }
            }

            // Winit builtin events
            winit::event::Event::RedrawRequested(window_id)
                if window_id == gpu_state.window().id() =>
            {
                match gpu_state.render() {
                    Ok(_) => {}
                    // The system is out of memory, we should probably quit
                    Err(wgpu::SurfaceError::OutOfMemory) => {
                        *control_flow = winit::event_loop::ControlFlow::Exit
                    }
                    // All other errors (Outdated, Timeout) should be resolved by the next frame
                    Err(e) => eprintln!("{:?}", e),
                }
            }
            winit::event::Event::MainEventsCleared => {}
            winit::event::Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == gpu_state.window().id() => {
                match event {
                    // Close
                    winit::event::WindowEvent::CloseRequested
                    | winit::event::WindowEvent::KeyboardInput {
                        input:
                            winit::event::KeyboardInput {
                                state: winit::event::ElementState::Pressed,
                                virtual_keycode: Some(winit::event::VirtualKeyCode::Escape),
                                ..
                            },
                        ..
                    } => *control_flow = winit::event_loop::ControlFlow::Exit,
                    _ => {}
                }
            }
            _ => {}
        }
    });
}
