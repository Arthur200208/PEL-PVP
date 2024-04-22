import peft
from MyModel import MyModel
from utils import xcl


def getMyPeftModel(model, r, a, dropout=0.5, m=5):
    # LORA by PEFT
    xcl.getPaparametersNum(model)
    target_modules = []
    for idx in range(m, 30):
        for layer_name in ["self_attn.q_proj", "self_attn.k_proj",
                           "self_attn.v_proj", "self_attn.out_proj"]:
            target_modules.append(f"layers.{idx}.{layer_name}")
    config = peft.LoraConfig(r=r, lora_alpha=a, target_modules=target_modules, lora_dropout=dropout)
    peft_esm2 = peft.get_peft_model(model, config)
    xcl.getPaparametersNum(peft_esm2)
    model = MyModel(peft_esm2)
    xcl.getPaparametersNum(model)
    return model
