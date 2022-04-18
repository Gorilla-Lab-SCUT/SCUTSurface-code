import torch
import torch.nn as nn
import torch.nn.functional as F

##############################################################
# Networks
##############################################################
class test_net(nn.Module):
	def __init__(self, point_dim, gf_dim):
		super(test_net, self).__init__()
		self.point_dim = point_dim
		self.gf_dim = gf_dim
		self.linear_1 = nn.Linear(self.point_dim, self.gf_dim*8, bias=True)
		self.linear_2 = nn.Linear(self.gf_dim*8, self.gf_dim*8, bias=True)
		self.linear_3 = nn.Linear(self.gf_dim*8, self.gf_dim*8, bias=True)
		self.linear_4 = nn.Linear(self.gf_dim*8, self.gf_dim*4, bias=True)
		self.linear_5 = nn.Linear(self.gf_dim*4, self.gf_dim*2, bias=True)
		self.linear_6 = nn.Linear(self.gf_dim*2, self.gf_dim, bias=True)

	def forward(self, points):
		l1 = self.linear_1(points)
		l1 = F.leaky_relu(l1, negative_slope=0.02, inplace=True)
		l2 = self.linear_2(l1)
		l2 = F.leaky_relu(l2, negative_slope=0.02, inplace=True)
		l3 = self.linear_3(l2)
		l3 = F.leaky_relu(l3, negative_slope=0.02, inplace=True)
		l4 = self.linear_4(l3)
		l4 = F.leaky_relu(l4, negative_slope=0.02, inplace=True)
		l5 = self.linear_5(l4)
		l5 = F.leaky_relu(l5, negative_slope=0.02, inplace=True)
		l6 = self.linear_6(l5).sum(axis=-2).squeeze()
		return l6


class test_net256(nn.Module):
	def __init__(self, point_dim, gf_dim):
		super(test_net256, self).__init__()
		self.point_dim = point_dim
		self.gf_dim = gf_dim
		self.linear_1 = nn.Linear(self.point_dim, self.gf_dim, bias=True)
		self.linear_2 = nn.Linear(self.gf_dim, self.gf_dim, bias=True)
		self.linear_3 = nn.Linear(self.gf_dim, self.gf_dim, bias=True)
		self.linear_4 = nn.Linear(self.gf_dim, self.gf_dim, bias=True)
		self.linear_5 = nn.Linear(self.gf_dim, self.gf_dim, bias=True)
		self.linear_6 = nn.Linear(self.gf_dim, self.gf_dim, bias=True)

	def forward(self, points):
		l1 = self.linear_1(points)
		l1 = F.leaky_relu(l1, negative_slope=0.02, inplace=True)
		l2 = self.linear_2(l1)
		l2 = F.leaky_relu(l2, negative_slope=0.02, inplace=True)
		l3 = self.linear_3(l2)
		l3 = F.leaky_relu(l3, negative_slope=0.02, inplace=True)
		l4 = self.linear_4(l3)
		l4 = F.leaky_relu(l4, negative_slope=0.02, inplace=True)
		l5 = self.linear_5(l4)
		l5 = F.leaky_relu(l5, negative_slope=0.02, inplace=True)
		l6 = self.linear_6(l5).sum(axis=-2).squeeze()
		return l6	