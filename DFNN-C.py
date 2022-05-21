import networkx as nx
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

Node = collections.namedtuple('Node', ['id', 'input', 'extras', 'type'])


def make_graph(node_num, p=0.75, k=4, m=5, graph_mode="WS"):
    if graph_mode == "ER":
        return nx.random_graphs.erdos_renyi_graph(node_num, p)
    elif graph_mode == "WS":
        return nx.random_graphs.connected_watts_strogatz_graph(
            node_num, k, p, tries=200)
    elif graph_mode == "BA":
        return nx.random_graphs.barabasi_albert_graph(node_num, m)
    else:
        return nx.path_graph(node_num)


def dfs(graph, source=0):
    dfs_nodes = list(nx.dfs_tree(graph, source=source))
    print(dfs_nodes)
    Nodes = [Node(source, -1, [], 1)]
    tmp_nodes = [source]
    for i in range(1, len(dfs_nodes)):
        node = dfs_nodes[i]
        tmp_nodes.append(node)
        input = dfs_nodes[i-1]
        nebors = list(graph.neighbors(node))
        extra = []
        for nebor in nebors:
            if nebor != input and nebor in tmp_nodes:
                extra.append(nebor)
        Nodes.append(Node(node, input, extra, 0))
    return Nodes, dfs_nodes[-1]


class depthwise_separable_conv_3x3(nn.Module):
    def __init__(self, nin, nout, stride):
        super(depthwise_separable_conv_3x3, self).__init__()
        self.depthwise = nn.Conv2d(
            nin, nin, kernel_size=3, stride=stride, padding=1, groups=nin)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class Triplet_unit(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1):
        super(Triplet_unit, self).__init__()
        self.relu = nn.ReLU()
        self.conv = depthwise_separable_conv_3x3(inplanes, outplanes, stride)
        self.bn = nn.BatchNorm2d(outplanes)

    def forward(self, x):
        out = self.relu(x)
        out = self.conv(out)
        out = self.bn(out)
        return out


class Node_OP(nn.Module):
    def __init__(self, Node, inplanes, outplanes):
        super(Node_OP, self).__init__()
        self.Node = Node
        self.is_input_node = Node.type == 1

        if self.is_input_node:
            self.unit_op = Triplet_unit(inplanes, outplanes, stride=2)
        else:
            self.unit_op = Triplet_unit(inplanes, outplanes, stride=1)

    def forward(self, input, extras=[]):
        for extra in extras:
            input = torch.cat([input, extra], dim=1)

        unit_out = self.unit_op(input)

        return unit_out


class StageBlock(nn.Module):
    def __init__(self, inplanes, outplanes, graph, source):
        super(StageBlock, self).__init__()
        self.nodes, self.last_node = dfs(graph, source)
        print(self.nodes, self.last_node)
        self.nodeop = nn.ModuleList()
        self.source = source

        for index, node in enumerate(self.nodes):
            if index == 0:
                self.nodeop.append(Node_OP(node, inplanes, outplanes))
            else:
                self.nodeop.append(
                    Node_OP(node, outplanes*(1+len(node.extras)), outplanes))

    def forward(self, x):
        results = {}
        for id, node in enumerate(self.nodes):
            if node.id == self.source:
                results[node.id] = self.nodeop[id](x)
            else:
                res = self.nodeop[id](results[node.input],
                                      [results[ex] for ex in node.extras])
                results[node.id] = res

        return results[self.last_node]


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.graph_mode = args.graph_mode
        self.num_classes = args.num_classes
        self.node_num = args.node_num
        self.p = args.p
        self.k = args.k
        self.m = args.m
        self.channel = args.c

        self.dropout_rate = 0.2

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.channel,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(self.channel),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.channel,
                      out_channels=self.channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.channel),
        )

        graph = make_graph(self.node_num, p=self.p, k=self.k,
                           m=self.m, graph_mode=self.graph_mode)
        self.conv3 = StageBlock(self.channel,
                                self.channel*2, graph, random.randint(0, self.node_num-1))

        self.conv4 = StageBlock(self.channel*2,
                                self.channel*4, graph, random.randint(0, self.node_num-1))
        self.classifier = nn.Sequential(
            nn.Conv2d(self.channel * 4, 1280, kernel_size=1),
            nn.BatchNorm2d(1280)
        )
        self.output = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(1280, self.num_classes)
        )

    def forward(self, x):

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.classifier(out)

        # global average pooling
        _, _, height, width = out.size()
        out = F.avg_pool2d(out, kernel_size=[height, width])
        # out = F.avg_pool2d(out, kernel_size=x.size()[2:])
        out = torch.squeeze(out)
        out = self.output(out)

        return out


def dfnn_c(args):
    return Model(args)
