import gradio as gr

# UI Interface
with gr.Blocks() as app_interface:
    
    # [0, 0]
    with gr.Row():
        
        # [0, 0]
        with gr.Column():
            input_src = gr.Image(label="Input", sources="webcam")
        # [0, 1]
        with gr.Column():
            output_video = gr.Image(label="Output")
            
        input_src.stream(lambda s: s, 
                         input_src, 
                         output_video, 
                         time_limit=15, 
                         stream_every=0.1, 
                         concurrency_limit=30)

    # [1, 0]
    with gr.Row():
        
        # [1, 0]
        with gr.Column():
            text_out = gr.Textbox(label="Real-time Output")


if __name__ == "__main__":
    app_interface.launch()
