from mechamodlearn.rigidbody import LearnedRigidBody
import torch
from datetime import datetime
from pathlib import Path

DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
class ExternalData:
    def __init__(self, qdim, udim, device):

        self._qdim = qdim
        self._udim = udim
        self._device = device
        self.thetamask = torch.ones(10, device=DEVICE)
        print('self.thetamask',self.thetamask)
        
def load_and_save_full_model(epoch: int, logdir: str):
    # 加载参数
    system = ExternalData(10, 10, DEVICE)
    model = LearnedRigidBody(system._qdim, system._udim, system.thetamask)
    model_path = Path(logdir) / f"model_state_epoch_{epoch}.th"
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # 保存完整模型
    full_model_path = Path(logdir) / f"full_model_epoch_{epoch}.pt"
    torch.save(model, full_model_path)  # 保存结构和参数
    # traced_script_module = torch.jit.script(model)
    # traced_script_module.save(full_model_path)
    
    return model

if __name__ == '__main__':
    load_and_save_full_model(599, 'simplelog/20250404_014428')
