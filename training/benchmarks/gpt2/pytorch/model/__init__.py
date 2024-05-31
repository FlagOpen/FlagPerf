from model.models.gpt_model import GPTModel

def create_model(args):
    # config.resume_step = 0
    
    model = GPTModel(
        num_tokentypes=0,
        parallel_output=True,
        pre_process=True,
        post_process=True,
    )

    return None, model,
