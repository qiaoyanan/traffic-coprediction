from train.train import train
from utils.load_config import get_Parameter

if __name__ == '__main__':
    print(
        f'current method:{get_Parameter("model_name")}|| current target:{get_Parameter("target")}|| current data:{get_Parameter("data_path")}||current config:{get_Parameter((get_Parameter("model_name"), "description"))}')
    model_result = train()
