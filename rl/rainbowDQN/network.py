import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math

class NoisyLinear(nn.Module):
    """Noisy linear module for NoisyNet.
    
    Attributes:
        in_features (int): input size of linear module
        out_features (int): output size of linear module
        std_init (float): initial std value
        weight_mu (nn.Parameter): mean value weight parameter
        weight_sigma (nn.Parameter): std value weight parameter
        bias_mu (nn.Parameter): mean value bias parameter
        bias_sigma (nn.Parameter): std value bias parameter
        
    """

    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        std_init: float = 0.5,
    ):
        """Initialization."""
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(
            torch.Tensor(out_features, in_features)
        )
        self.register_buffer(
            "weight_epsilon", torch.Tensor(out_features, in_features)
        )

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Reset trainable network parameters (factorized gaussian noise)."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.in_features)
        )
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(
            self.std_init / math.sqrt(self.out_features)
        )

    def reset_noise(self):
        """Make new noise."""
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        # outer product
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation.
        
        We don't use separate statements on train / eval mode.
        It doesn't show remarkable difference of performance.
        """
        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )
    
    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        """Set scale to make noise (factorized gaussian noise)."""
        x = torch.randn(size)

        return x.sign().mul(x.abs().sqrt())
    

class ImageBackbone(nn.Module):
    def __init__(self, out_dim):
        super(ImageBackbone, self).__init__()
        #self.resnet18 = models.resnet18(pretrained=True)
        #self.resnet18.fc = nn.Linear(512, out_dim)  # Modify the fully connected layer for your desired number of classes

        #self.backbone = models.mobilenet_v2(pretrained=True)
        #in_features = self.backbone.classifier[-1].in_features  # Get the number of input features to the last layer
        #self.backbone.classifier[-1] = nn.Linear(in_features, out_dim)

        self.backbone = models.squeezenet1_0(pretrained=True)
        self.backbone.classifier[1] = nn.Conv2d(512, out_dim, kernel_size=(1, 1), stride=(1, 1))  # Adjust to your 'out_dim'

    def forward(self, x):
        x = self.backbone(x)
        return x

class Network(nn.Module):
    def __init__(
        self, 
        in_dim: int,
        out_dim: int,
        atom_size: int, 
        support: torch.Tensor,
        hidden_dim: int = 128,
        use_state = True
    ):
        """Initialization."""
        super(Network, self).__init__()
        
        self.support = support
        self.out_dim = out_dim
        self.atom_size = atom_size
        self.use_state = use_state

        # set common feature layer
        self.feature_layer = nn.Sequential(
            #nn.Linear(in_dim, 128),
            #nn.ReLU(),
            ImageBackbone(out_dim=hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.qvel_feature_layer = nn.Sequential(
            nn.Linear(14, hidden_dim),
            nn.ReLU()
        )

        self.all_feature_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU()
        )
       
        if self.use_state:
            self.state_layer = nn.Sequential(
                nn.Linear(39 + 14, hidden_dim),
                nn.ReLU()
            )

        # set advantage layer
        self.advantage_hidden_layer = NoisyLinear(hidden_dim * 2, hidden_dim)
        self.advantage_layer = NoisyLinear(hidden_dim, out_dim * atom_size)

        # set value layer
        self.value_hidden_layer = NoisyLinear(hidden_dim * 2, hidden_dim)
        self.value_layer = NoisyLinear(hidden_dim, atom_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2)
        
        return q
    
    def dist(self, x: torch.Tensor) -> torch.Tensor:
        """Get distribution for atoms."""

        if not self.use_state:
            obs = x[:, :-14]
            qvel = x[:, -14:]
            x = obs.reshape([-1, 96, 128, 3]).permute(0, 3, 1, 2)

            obs_feature = self.feature_layer(x)
            qvel_feature = self.qvel_feature_layer(qvel)

            feature = torch.concatenate((obs_feature, qvel_feature), dim=1)
        else:
            state = x[:, :-14]
            qvel = x[:, -14:]

            state_feature = self.state_layer(state)
            qvel_feature = self.qvel_feature_layer(qvel)

            feature = torch.concatenate((state_feature, qvel_feature), dim=1)
        
        feature = self.all_feature_layer(feature)

        adv_hid = F.relu(self.advantage_hidden_layer(feature))
        val_hid = F.relu(self.value_hidden_layer(feature))
        
        advantage = self.advantage_layer(adv_hid).view(
            -1, self.out_dim, self.atom_size
        )
        value = self.value_layer(val_hid).view(-1, 1, self.atom_size)
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # for avoiding nans
        
        return dist
    
    def reset_noise(self):
        """Reset all noisy layers."""
        self.advantage_hidden_layer.reset_noise()
        self.advantage_layer.reset_noise()
        self.value_hidden_layer.reset_noise()
        self.value_layer.reset_noise()
