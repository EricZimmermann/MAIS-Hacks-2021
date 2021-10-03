import gradio as gr
import argparse

def process(lang_index, prompt_type, sketch):
    # Init Outputs
    instructions = None
    visual_prompt = None
    audio_prompt = None
    label_output = None
    interpret_image = None
    latent_video = None

    # Set instructions
    instructions = "Draw the character according to the prompt below."

    return instructions, visual_prompt, audio_prompt, label_output, interpret_image, latent_video

def add_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--port', type=int, required=True, help='port to launch app')
    return parser

def main():
    parser = add_argparser()
    args = parser.parse_args()

    # Inputs
    dropdown_menu = gr.inputs.Dropdown(["English",
                                        "Greek",
                                        "Hebrew",
                                        "Japanese Katakana",
                                        "Japanese Hiragana"],
                                       type="index",
                                       label="Character Set"
                                      )
    radio_choice = gr.inputs.Radio(["Audio","Visual","Text"],
                                   type="index",
                                   label="Prompt type"
                                  )
    sketchpad = gr.inputs.Image((32,32),
                                image_mode="L",
                                invert_colors=True,
                                source="canvas",
                                label="Canvas"
                               )

    # Outputs
    instruction_textbox = gr.outputs.Textbox(label="Instructions")
    visual_prompt = gr.outputs.Image(label="Visual Prompt")
    audio_prompt = gr.outputs.Audio(label="Audio Prompt")
    label_output = gr.outputs.Label(num_top_classes=5,
                                    label="Predicted character"
                                   )
    interpret_image = gr.outputs.Image(label="Interpretation")
    latent_video = gr.outputs.HTML(label="Latent Interpolation")


    interface = gr.Interface(fn=process, 
                             inputs=[dropdown_menu,
                                     radio_choice,
                                     sketchpad,
                                    ],
                             outputs=[instruction_textbox,
                                      visual_prompt,
                                      audio_prompt,
                                      label_output,
                                      interpret_image,
                                      latent_video,
                                     ],
                             live=False, server_port=args.port)
    interface.launch(inline=False, inbrowser=False)

if __name__ == "__main__":
    main()
