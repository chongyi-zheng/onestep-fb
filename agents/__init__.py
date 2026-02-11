from agents.byol_gamma import BYOLGammaAgent
from agents.fb import FBAgent
from agents.hilp import HILPAgent
from agents.icvf import ICVFAgent
from agents.laplacian import LaplacianAgent
from agents.onestep_fb import OneStepFBAgent

agents = dict(
    byol_gamma=BYOLGammaAgent,
    fb=FBAgent,
    hilp=HILPAgent,
    icvf=ICVFAgent,
    laplacian=LaplacianAgent,
    onestep_fb=OneStepFBAgent,
)
