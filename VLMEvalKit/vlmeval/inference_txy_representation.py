import torch
import torch.distributed as dist
from vlmeval.config import supported_VLM
from vlmeval.utils import track_progress_rich
from vlmeval.smp import *
import sys
from accelerate import Accelerator
from PIL import Image

sys.path.insert(0, "/home/zhanyuliang/Project/CoT/GLoRE")
# 添加 Qwen_Math 所在 src 路径
sys.path.insert(0, "/home/zhanyuliang/Project/CoT/GLoRE/src")

from src.utils.utils import read_jsonl, write_jsonl, transformodel_name2model_path, load_model_tokenizer, get_model_wrapper
import pdb
FAIL_MSG = 'Failed to obtain answer via API.'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, nargs='+', required=True)
    parser.add_argument('--model', type=str, nargs='+', required=True)
    parser.add_argument('--nproc', type=int, default=4, required=True)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    return args

def inject(model_name,model,args):
    # 注入特征
    long_form_demon_path = "/mnt2/zhanyuliang/CoT/GLoRE/Dataset/STILL/data/long_short_form_thought.jsonl"
    demon_data = read_jsonl(long_form_demon_path)  
    accelerator = Accelerator()
    if "llava_next_merge" in model_name:
        tokenizer = model.processor.tokenizer
        model_config = model.model.config
        model_wrapper = get_model_wrapper(model_name, model, tokenizer, model_config, accelerator)
        device = accelerator.device
        extract_pos = "last"
        style_inject_method = "linear"
        style_inject_pos = "first"
        lambda_inj = args.lambda_inj
        style_strength = f"{lambda_inj},1.0"
        module = "hidden"
        style_config = {
            "tok_pos": extract_pos,
            "inject_method": style_inject_method,
            "inject_pos": style_inject_pos,
            "strength": style_strength,
            'module': module,
        }
        layer = args.inj_layer
        short_form_vector_list = []
        long_form_vector_list = []
        for i, d in tqdm(enumerate(demon_data), desc="get vector"):
            problem = d['problem']
            short_form = d['short_form_thought']
            long_form = d['long_form_thought']
            
            # ========== process short form demonstration ==========
            short_form_demon_list = [
                {"role": "user", "content": (problem).strip()},
                {"role": "assistant", "content": short_form}
            ]
            short_form_demon_str = tokenizer.apply_chat_template(short_form_demon_list, tokenize=False)    
            
            short_form_all_latent_dicts = []
            with torch.no_grad():
                with model_wrapper.extract_latent():
                    short_form_demon_token = tokenizer(short_form_demon_str, return_tensors='pt').to(device)
                    _ = model.model.language_model(**short_form_demon_token)
                short_form_all_latent_dicts.append(model_wrapper.latent_dict)
            
            short_form_context_vector_dict = model_wrapper.get_context_vector(short_form_all_latent_dicts, style_config)
            short_form_context_vector_dict = {key: value for key, value in short_form_context_vector_dict.items() if int(key) == layer}
            short_form_vector_list.append(short_form_context_vector_dict)
            del short_form_all_latent_dicts
            
            # ========== process long form demonstration ==========
            long_form_demon_list = [
                {"role": "user", "content": (problem).strip()},
                {"role": "assistant", "content": long_form}
            ]
            long_form_demon_str = tokenizer.apply_chat_template(long_form_demon_list, tokenize=False)
            
            long_form_all_latent_dicts = []
            with torch.no_grad():
                with model_wrapper.extract_latent():
                    long_form_demon_token = tokenizer(long_form_demon_str, return_tensors='pt').to(device)
                    _ = model.model.language_model(**long_form_demon_token)
                long_form_all_latent_dicts.append(model_wrapper.latent_dict)
                
            long_form_context_vector_dict = model_wrapper.get_context_vector(long_form_all_latent_dicts, style_config)
            long_form_context_vector_dict = {key: value for key, value in long_form_context_vector_dict.items() if int(key) == layer}
            long_form_vector_list.append(long_form_context_vector_dict)
            del long_form_all_latent_dicts
        # ========== get contrast vector ==========
        
        contrast_vector_list = []
        aggregated_contrast_vector_dict = defaultdict(lambda: defaultdict(torch.Tensor))
        
        for i, (short_form, long_form) in enumerate(zip(short_form_vector_list, long_form_vector_list)):
            demon_contrast_vector_dict = defaultdict(lambda: defaultdict(torch.Tensor))
            module = style_config['module']
            short_form_layer = short_form[layer][module]
            long_form_layer = long_form[layer][module]
            contrast_vector = long_form_layer - short_form_layer
            demon_contrast_vector_dict[layer][module] = contrast_vector
            contrast_vector_list.append(demon_contrast_vector_dict)

        for demon_contrast_vector_dict in contrast_vector_list:
            for module in demon_contrast_vector_dict[layer]:
                contrast_vector = demon_contrast_vector_dict[layer][module]
                
                if aggregated_contrast_vector_dict[layer][module].numel() == 0:
                    aggregated_contrast_vector_dict[layer][module] = contrast_vector
                else:
                    aggregated_contrast_vector_dict[layer][module] += contrast_vector

        num_demons = len(contrast_vector_list)
        for module in aggregated_contrast_vector_dict[layer]:
            aggregated_contrast_vector_dict[layer][module] /= num_demons
        
        del contrast_vector_list, short_form_vector_list, long_form_vector_list
    elif "llava" in model_name:
        tokenizer = model.processor.tokenizer
        model_config = model.model.config
        model_wrapper = get_model_wrapper(model_name, model, tokenizer, model_config, accelerator)
        device = accelerator.device
        extract_pos = "last"
        style_inject_method = "linear"
        style_inject_pos = "first"
        lambda_inj = args.lambda_inj
        style_strength = f"{lambda_inj},1.0"
        module = "hidden"
        style_config = {
            "tok_pos": extract_pos,
            "inject_method": style_inject_method,
            "inject_pos": style_inject_pos,
            "strength": style_strength,
            'module': module,
        }
        layer = args.inj_layer
        short_form_vector_list = []
        long_form_vector_list = []
        for i, d in tqdm(enumerate(demon_data), desc="get vector"):
            problem = d['problem']
            short_form = d['short_form_thought']
            long_form = d['long_form_thought']
            
            # ========== process short form demonstration ==========
            short_form_demon_list = [
                {"role": "user", "content": [{"type": "text", "text": problem.strip()}]},
                {"role": "assistant", "content": [{"type": "text", "text": short_form}]}
            ]
            short_form_demon_str = model.processor.apply_chat_template(short_form_demon_list, tokenize=False)
            
            short_form_all_latent_dicts = []
            with torch.no_grad():
                with model_wrapper.extract_latent():
                    short_form_demon_token = tokenizer(short_form_demon_str, return_tensors='pt').to(device)
                    _ = model.model.language_model(**short_form_demon_token)
                short_form_all_latent_dicts.append(model_wrapper.latent_dict)
            
            short_form_context_vector_dict = model_wrapper.get_context_vector(short_form_all_latent_dicts, style_config)
            short_form_context_vector_dict = {key: value for key, value in short_form_context_vector_dict.items() if int(key) == layer}
            short_form_vector_list.append(short_form_context_vector_dict)
            del short_form_all_latent_dicts
            
            # ========== process long form demonstration ==========
            long_form_demon_list = [
                {"role": "user", "content": [{"type": "text", "text": problem.strip()}]},
                {"role": "assistant", "content": [{"type": "text", "text": long_form}]}
            ]
            long_form_demon_str = model.processor.apply_chat_template(long_form_demon_list, tokenize=False)

            
            long_form_all_latent_dicts = []
            with torch.no_grad():
                with model_wrapper.extract_latent():
                    long_form_demon_token = tokenizer(long_form_demon_str, return_tensors='pt').to(device)
                    _ = model.model.language_model(**long_form_demon_token)
                long_form_all_latent_dicts.append(model_wrapper.latent_dict)
                
            long_form_context_vector_dict = model_wrapper.get_context_vector(long_form_all_latent_dicts, style_config)
            long_form_context_vector_dict = {key: value for key, value in long_form_context_vector_dict.items() if int(key) == layer}
            long_form_vector_list.append(long_form_context_vector_dict)
            del long_form_all_latent_dicts
        # ========== get contrast vector ==========
        
        contrast_vector_list = []
        aggregated_contrast_vector_dict = defaultdict(lambda: defaultdict(torch.Tensor))
        
        for i, (short_form, long_form) in enumerate(zip(short_form_vector_list, long_form_vector_list)):
            demon_contrast_vector_dict = defaultdict(lambda: defaultdict(torch.Tensor))
            module = style_config['module']
            short_form_layer = short_form[layer][module]
            long_form_layer = long_form[layer][module]
            contrast_vector = long_form_layer - short_form_layer
            demon_contrast_vector_dict[layer][module] = contrast_vector
            contrast_vector_list.append(demon_contrast_vector_dict)

        for demon_contrast_vector_dict in contrast_vector_list:
            for module in demon_contrast_vector_dict[layer]:
                contrast_vector = demon_contrast_vector_dict[layer][module]
                
                if aggregated_contrast_vector_dict[layer][module].numel() == 0:
                    aggregated_contrast_vector_dict[layer][module] = contrast_vector
                else:
                    aggregated_contrast_vector_dict[layer][module] += contrast_vector

        num_demons = len(contrast_vector_list)
        for module in aggregated_contrast_vector_dict[layer]:
            aggregated_contrast_vector_dict[layer][module] /= num_demons
        
        del contrast_vector_list, short_form_vector_list, long_form_vector_list
    elif "Qwen2" in model_name:
        tokenizer = model.processor.tokenizer
        model_config = model.model.config
        model_wrapper = get_model_wrapper(model_name, model.model, tokenizer, model_config, accelerator)
        device = accelerator.device
        extract_pos = "last"
        style_inject_method = "linear"
        style_inject_pos = "first"
        lambda_inj = args.lambda_inj
        style_strength = f"{lambda_inj},1.0"
        module = "hidden"
        style_config = {
            "tok_pos": extract_pos,
            "inject_method": style_inject_method,
            "inject_pos": style_inject_pos,
            "strength": style_strength,
            'module': module,
        }
        layer = args.inj_layer
        short_form_vector_list = []
        long_form_vector_list = []
        for i, d in tqdm(enumerate(demon_data), desc="get vector"):
            problem = d['problem']
            short_form = d['short_form_thought']
            long_form = d['long_form_thought']
            
            # ========== process short form demonstration ==========
            short_form_demon_list = [
                {"role": "user", "content": [{"type": "text", "text": problem.strip()}]},
                {"role": "assistant", "content": [{"type": "text", "text": short_form}]}
            ]
            short_form_demon_str = model.processor.apply_chat_template(short_form_demon_list, tokenize=False)
            
            short_form_all_latent_dicts = []
            with torch.no_grad():
                with model_wrapper.extract_latent():
                    short_form_demon_token = tokenizer(short_form_demon_str, return_tensors='pt').to(device)
                    _ = model.model.model(**short_form_demon_token)
                short_form_all_latent_dicts.append(model_wrapper.latent_dict)
            
            short_form_context_vector_dict = model_wrapper.get_context_vector(short_form_all_latent_dicts, style_config)
            short_form_context_vector_dict = {key: value for key, value in short_form_context_vector_dict.items() if int(key) == layer}
            short_form_vector_list.append(short_form_context_vector_dict)
            del short_form_all_latent_dicts
            
            # ========== process long form demonstration ==========
            long_form_demon_list = [
                {"role": "user", "content": [{"type": "text", "text": problem.strip()}]},
                {"role": "assistant", "content": [{"type": "text", "text": long_form}]}
            ]
            long_form_demon_str = model.processor.apply_chat_template(long_form_demon_list, tokenize=False)

            
            long_form_all_latent_dicts = []
            with torch.no_grad():
                with model_wrapper.extract_latent():
                    long_form_demon_token = tokenizer(long_form_demon_str, return_tensors='pt').to(device)
                    _ = model.model.model(**long_form_demon_token)
                long_form_all_latent_dicts.append(model_wrapper.latent_dict)
                
            long_form_context_vector_dict = model_wrapper.get_context_vector(long_form_all_latent_dicts, style_config)
            long_form_context_vector_dict = {key: value for key, value in long_form_context_vector_dict.items() if int(key) == layer}
            long_form_vector_list.append(long_form_context_vector_dict)
            del long_form_all_latent_dicts
        # ========== get contrast vector ==========
        
        contrast_vector_list = []
        aggregated_contrast_vector_dict = defaultdict(lambda: defaultdict(torch.Tensor))
        
        for i, (short_form, long_form) in enumerate(zip(short_form_vector_list, long_form_vector_list)):
            demon_contrast_vector_dict = defaultdict(lambda: defaultdict(torch.Tensor))
            module = style_config['module']
            short_form_layer = short_form[layer][module]
            long_form_layer = long_form[layer][module]
            contrast_vector = long_form_layer - short_form_layer
            demon_contrast_vector_dict[layer][module] = contrast_vector
            contrast_vector_list.append(demon_contrast_vector_dict)

        for demon_contrast_vector_dict in contrast_vector_list:
            for module in demon_contrast_vector_dict[layer]:
                contrast_vector = demon_contrast_vector_dict[layer][module]
                
                if aggregated_contrast_vector_dict[layer][module].numel() == 0:
                    aggregated_contrast_vector_dict[layer][module] = contrast_vector
                else:
                    aggregated_contrast_vector_dict[layer][module] += contrast_vector

        num_demons = len(contrast_vector_list)
        for module in aggregated_contrast_vector_dict[layer]:
            aggregated_contrast_vector_dict[layer][module] /= num_demons
        
        del contrast_vector_list, short_form_vector_list, long_form_vector_list
    elif "InternVL2" in model_name:
        tokenizer = model.tokenizer
        model_config = model.model.config
        model_wrapper = get_model_wrapper(model_name, model.model, tokenizer, model_config, accelerator)
        device = accelerator.device
        extract_pos = "last"
        style_inject_method = "linear"
        style_inject_pos = "first"
        lambda_inj = args.lambda_inj
        style_strength = f"{lambda_inj},1.0"
        module = "hidden"
        style_config = {
            "tok_pos": extract_pos,
            "inject_method": style_inject_method,
            "inject_pos": style_inject_pos,
            "strength": style_strength,
            'module': module,
        }
        layer = args.inj_layer
        short_form_vector_list = []
        long_form_vector_list = []
        for i, d in tqdm(enumerate(demon_data), desc="get vector"):
            problem = d['problem']
            short_form = d['short_form_thought']
            long_form = d['long_form_thought']
            
            # ========== process short form demonstration ==========
            short_form_demon_list = [
                {"role": "user", "content": (problem).strip()},
                {"role": "assistant", "content": short_form}
            ]
            short_form_demon_str = tokenizer.apply_chat_template(short_form_demon_list, tokenize=False)    
            
            short_form_all_latent_dicts = []
            with torch.no_grad():
                with model_wrapper.extract_latent():
                    short_form_demon_token = tokenizer(short_form_demon_str, return_tensors='pt')
                    short_form_demon_token = {k: v.to(device) for k, v in short_form_demon_token.items()}
                    _ = model.model.language_model(**short_form_demon_token)
                short_form_all_latent_dicts.append(model_wrapper.latent_dict)
            
            short_form_context_vector_dict = model_wrapper.get_context_vector(short_form_all_latent_dicts, style_config)
            short_form_context_vector_dict = {key: value for key, value in short_form_context_vector_dict.items() if int(key) == layer}
            short_form_vector_list.append(short_form_context_vector_dict)
            del short_form_all_latent_dicts
            
            # ========== process long form demonstration ==========
            long_form_demon_list = [
                {"role": "user", "content": (problem).strip()},
                {"role": "assistant", "content": long_form}
            ]
            long_form_demon_str = tokenizer.apply_chat_template(long_form_demon_list, tokenize=False)
            
            long_form_all_latent_dicts = []
            with torch.no_grad():
                with model_wrapper.extract_latent():
                    long_form_demon_token = tokenizer(long_form_demon_str, return_tensors='pt')
                    long_form_demon_token = {k: v.to(device) for k, v in long_form_demon_token.items()}
                    _ = model.model.language_model(**long_form_demon_token)
                long_form_all_latent_dicts.append(model_wrapper.latent_dict)
                
            long_form_context_vector_dict = model_wrapper.get_context_vector(long_form_all_latent_dicts, style_config)
            long_form_context_vector_dict = {key: value for key, value in long_form_context_vector_dict.items() if int(key) == layer}
            long_form_vector_list.append(long_form_context_vector_dict)
            del long_form_all_latent_dicts
        # ========== get contrast vector ==========
        
        contrast_vector_list = []
        aggregated_contrast_vector_dict = defaultdict(lambda: defaultdict(torch.Tensor))
        
        for i, (short_form, long_form) in enumerate(zip(short_form_vector_list, long_form_vector_list)):
            demon_contrast_vector_dict = defaultdict(lambda: defaultdict(torch.Tensor))
            module = style_config['module']
            short_form_layer = short_form[layer][module]
            long_form_layer = long_form[layer][module]
            contrast_vector = long_form_layer - short_form_layer
            demon_contrast_vector_dict[layer][module] = contrast_vector
            contrast_vector_list.append(demon_contrast_vector_dict)

        for demon_contrast_vector_dict in contrast_vector_list:
            for module in demon_contrast_vector_dict[layer]:
                contrast_vector = demon_contrast_vector_dict[layer][module]
                
                if aggregated_contrast_vector_dict[layer][module].numel() == 0:
                    aggregated_contrast_vector_dict[layer][module] = contrast_vector
                else:
                    aggregated_contrast_vector_dict[layer][module] += contrast_vector

        num_demons = len(contrast_vector_list)
        for module in aggregated_contrast_vector_dict[layer]:
            aggregated_contrast_vector_dict[layer][module] /= num_demons
        
        del contrast_vector_list, short_form_vector_list, long_form_vector_list
    style_strength = [float(s) for s in style_config["strength"].split(',')]
    style_config["strength"] = style_strength
    return model_wrapper,aggregated_contrast_vector_dict,layer,style_config
# Only API model is accepted
def infer_data_api(model, work_dir, model_name, dataset, index_set=None, api_nproc=4, ignore_failed=False):
    rank, world_size = get_rank_and_world_size()
    assert rank == 0 and world_size == 1
    dataset_name = dataset.dataset_name
    data = dataset.data
    if index_set is not None:
        data = data[data['index'].isin(index_set)]

    model = supported_VLM[model_name]() if isinstance(model, str) else model
    assert getattr(model, 'is_api', False)
    if hasattr(model, 'set_dump_image'):
        model.set_dump_image(dataset.dump_image)

    lt, indices = len(data), list(data['index'])

    structs = []
    for i in range(lt):
        item = data.iloc[i]
        if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):
            assert hasattr(model, 'build_prompt')
            struct = model.build_prompt(item, dataset=dataset_name)
        else:
            struct = dataset.build_prompt(item)
        structs.append(struct)

    # structs = [dataset.build_prompt(data.iloc[i]) for i in range(lt)]

    out_file = f'{work_dir}/{model_name}_{dataset_name}_supp.pkl'
    res = {}
    if osp.exists(out_file):
        res = load(out_file)
        if ignore_failed:
            res = {k: v for k, v in res.items() if FAIL_MSG not in v}

    structs = [s for i, s in zip(indices, structs) if i not in res]
    indices = [i for i in indices if i not in res]

    gen_func = model.generate
    structs = [dict(message=struct, dataset=dataset_name) for struct in structs]

    if len(structs):
        track_progress_rich(gen_func, structs, nproc=api_nproc, chunksize=api_nproc, save=out_file, keys=indices)

    res = load(out_file)
    if index_set is not None:
        res = {k: v for k, v in res.items() if k in index_set}
    os.remove(out_file)
    return res
import torch

def modify_params(model, layer_num, module_type):
    # 获取对应层的参数路径
    layer_prefix = f"language_model.model.layers.{layer_num}"
    
    # 定义模块名和对应的参数
    module_params = {
        'self_attn': [
            f"{layer_prefix}.self_attn.q_proj.weight",
            f"{layer_prefix}.self_attn.k_proj.weight",
            f"{layer_prefix}.self_attn.v_proj.weight",
            f"{layer_prefix}.self_attn.o_proj.weight"
        ],
        'mlp': [
            f"{layer_prefix}.mlp.gate_proj.weight",
            f"{layer_prefix}.mlp.up_proj.weight",
            f"{layer_prefix}.mlp.down_proj.weight"
        ],
        # 'self_attn': [
            
        #     f"{layer_prefix}.self_attn.v_proj.weight",
           
        # ],
        # 'mlp': [
        #     f"{layer_prefix}.mlp.down_proj.weight"
        # ]
    }
    
    # 获取模块的参数
    params_to_modify = module_params.get(module_type)
    if params_to_modify is None:
        raise ValueError("Invalid module type. Choose 'self_attn' or 'mlp'.")
    
    # 遍历并修改参数
    for param_name in params_to_modify:
        if param_name in model.model.state_dict():
            param = model.model.state_dict()[param_name]
            pdb.set_trace()
            new_param = torch.ones_like(param)  # 创建一个新的参数，形状和原来一致，值都为1
            len=param.size()[0]
            new_param=new_param/len
            model.model.state_dict()[param_name].copy_(new_param)  # 更新模型参数
        else:
            print(f"Warning: {param_name} not found in the model state_dict.")
    
    return model


def infer_data(model, model_name, work_dir, dataset, out_file, verbose=False, api_nproc=4,merge_model=None,cut_layer=None,cut_module=None,args = None):
    dataset_name = dataset.dataset_name
    prev_file = f'{work_dir}/{model_name}_{dataset_name}_PREV.pkl'
    res = load(prev_file) if osp.exists(prev_file) else {}
    if osp.exists(out_file):
        res.update(load(out_file))

    rank, world_size = get_rank_and_world_size()
    sheet_indices = list(range(rank, len(dataset), world_size))
    lt = len(sheet_indices)
    data = dataset.data.iloc[sheet_indices]
    data_indices = [i for i in data['index']]

    # If finished, will exit without building the model
    all_finished = True
    for i in range(lt):
        idx = data.iloc[i]['index']
        if idx not in res:
            all_finished = False
    if all_finished:
        res = {k: res[k] for k in data_indices}
        dump(res, out_file)
        return

    # Data need to be inferred
    data = data[~data['index'].isin(res)]
    lt = len(data)

    model = supported_VLM[model_name]() if isinstance(model, str) else model
    if cut_layer!=None and cut_module!=None:
        model = modify_params(model, cut_layer, cut_module)

    if merge_model: 
        if 'llava' in model_name or 'intern' in model_name.lower():
            # Load the saved state_dict
            # 
            # merged_state_dict = torch.load(merge_model)
            # model.model.language_model.load_state_dict(merged_state_dict)
            # Print the device information for each parameter in the original model
            #  Print the device information for each parameter in the original model
            device_map = {}
            for name, param in model.model.language_model.named_parameters():
                device_map[name] = param.device  # Record the device for each parameter

            # Load the merged model's state dictionary directly to CUDA
            merged_state_dict = torch.load(merge_model, map_location='cpu')

            # Create a new state dictionary to match the original model's device distribution
            new_state_dict = {}
            for name, param in merged_state_dict.items():
                if name in device_map:
                    new_state_dict[name] = param.to(device_map[name])  # Move parameter to the corresponding device
                else:
                    print(f'Warning: Parameter {name} not found in original model.')

            # Load the new state dictionary into the language_model
            model.model.language_model.load_state_dict(new_state_dict)

            # Optionally, release any unnecessary memory
            del merged_state_dict  # Release the original merged state dict from memory
            torch.cuda.empty_cache()  # Clear CUDA cache
        elif 'Qwen' in model_name:
            # Load the saved state_dict
            merged_state_dict = torch.load(merge_model)
            model.model.model.load_state_dict(merged_state_dict)
        elif 'idefics' in model_name:
            merged_state_dict = torch.load(merge_model)
            # 
            model.model.model.text_model.load_state_dict(merged_state_dict)
    # ################################################################################################################################################################
    model_wrapper,aggregated_contrast_vector_dict,layer,style_config = inject(model_name,model,args)
    # ################################################################################################################################################################
    is_api = getattr(model, 'is_api', False)
    if is_api:
        lt, indices = len(data), list(data['index'])
        supp = infer_data_api(
            model=model,
            work_dir=work_dir,
            model_name=model_name,
            dataset=dataset,
            index_set=set(indices),
            api_nproc=api_nproc)
        for idx in indices:
            assert idx in supp
        res.update(supp)
        res = {k: res[k] for k in data_indices}
        dump(res, out_file)
        return model
    else:
        model.set_dump_image(dataset.dump_image)
    
    for i in tqdm(range(lt)):
        idx = data.iloc[i]['index']
        if idx in res:
            continue

        if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):
            struct = model.build_prompt(data.iloc[i], dataset=dataset_name)
        else:
            struct = dataset.build_prompt(data.iloc[i]) 
        with model_wrapper.inject_latent(aggregated_contrast_vector_dict, [layer], style_config):
            response = model.generate(message=struct, dataset=dataset_name)
        # response = model.generate(message=struct, dataset=dataset_name)
        # 
        # print('hah')
        if "InternVL2" in model_name:
            print("question: "+struct[0]['value'])
        elif "llava_next_merge" in model_name:
            print("question: "+struct[1]['value'])
        print("response: "+response)
        torch.cuda.empty_cache()

        if verbose:
            print(response, flush=True)

        res[idx] = response
        if (i + 1) % 10 == 0:
            dump(res, out_file)

    res = {k: res[k] for k in data_indices}
    dump(res, out_file)
    return model


# A wrapper for infer_data, do the pre & post processing
def infer_data_job(model, work_dir, model_name, dataset, verbose=False, api_nproc=4, ignore_failed=False,merge_model=None,cut_layer=None,cut_module=None,args=None):
    rank, world_size = get_rank_and_world_size()
    dataset_name = dataset.dataset_name
    result_file = osp.join(work_dir, f'{model_name}_{dataset_name}.xlsx')

    prev_file = f'{work_dir}/{model_name}_{dataset_name}_PREV.pkl'
    if osp.exists(result_file):
        if rank == 0:
            data = load(result_file)
            results = {k: v for k, v in zip(data['index'], data['prediction'])}
            if not ignore_failed:
                results = {k: v for k, v in results.items() if FAIL_MSG not in str(v)}
            dump(results, prev_file)
        if world_size > 1:
            dist.barrier()

    tmpl = osp.join(work_dir, '{}' + f'{world_size}_{dataset_name}.pkl')
    out_file = tmpl.format(rank)
    
    
    model = infer_data(
        model=model, work_dir=work_dir, model_name=model_name, dataset=dataset,
        out_file=out_file, verbose=verbose, api_nproc=api_nproc,merge_model=merge_model,cut_layer=cut_layer,cut_module=cut_module,args = args)
    if world_size > 1:
        dist.barrier()

    if rank == 0:
        data_all = {}
        for i in range(world_size):
            data_all.update(load(tmpl.format(i)))

        data = dataset.data
        for x in data['index']:
            assert x in data_all
        data['prediction'] = [str(data_all[x]) for x in data['index']]
        if 'answer' in data.columns:
            data['ground_truth'] = data['answer']
        elif 'label' in data.columns:
            data['ground_truth'] = data['label']
        elif 'gt_answer' in data.columns:
            data['ground_truth'] = data['gt_answer']
        else:
            print("⚠️ Warning: No known ground-truth column found (e.g., answer/label/gt_answer)")
        if 'image' in data:
            data.pop('image')

        dump(data, result_file)
        for i in range(world_size):
            os.remove(tmpl.format(i))
    if world_size > 1:
        dist.barrier()
    return model
