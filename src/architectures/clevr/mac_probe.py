"""
MIT License

Copyright (c) 2018 Kim Seonghyeon

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
from torch.autograd import Variable
from torch import nn
from torch.nn.init import kaiming_uniform_, xavier_uniform_, normal
import torch.nn.functional as F

def linear(in_dim, out_dim, bias=True):
    lin = nn.Linear(in_dim, out_dim, bias=bias)
    xavier_uniform_(lin.weight)
    if bias:
        lin.bias.data.zero_()

    return lin

class ControlUnit(nn.Module):
    def __init__(self, dim, max_step, ncf=0):
        super().__init__()

        #import pdb
        #pdb.set_trace()

        self.position_aware = nn.ModuleList()
        for i in range(max_step):
            self.position_aware.append(linear(dim * 2 + ncf, dim))

        self.control_question = linear(dim * 2, dim)
        self.attn = linear(dim, 1)

        self.dim = dim

    def forward(self, step, context, question, control):

        # context: (bs, seq_len, 64)
        # question: (bs, 128)
        # control: (bs, 64)

        # (bs, 64)
        # this is q_i, computed from q
        position_aware = self.position_aware[step](question)

        # cq: (bs, 128)
        control_question = torch.cat([control, position_aware], 1)
        # cq: (8, 64)
        control_question = self.control_question(control_question)
        # cq: (bs, 1, 64)
        control_question = control_question.unsqueeze(1)

        # (bs, 1, 64) * (bs, seq_len, 64)
        # = (bs, seq_len, 64)
        context_prod = control_question * context
        # (bs, seq_len, 1)
        attn_weight = self.attn(context_prod)
        # (bs, seq_len, 1)
        attn = F.softmax(attn_weight, 1)

        # (bs seq_len, 64).sum(1)
        # = (bs, 64)
        next_control = (attn * context).sum(1)

        return next_control


class ReadUnit(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.mem = linear(dim, dim)
        self.concat = linear(dim * 2, dim)
        self.attn = linear(dim, 1)

    def forward(self, memory, know, control):
        mem = self.mem(memory[-1]).unsqueeze(2)
        concat = self.concat(torch.cat([mem * know, know], 1) \
                             .permute(0, 2, 1))
        attn = concat * control[-1].unsqueeze(1)
        attn = self.attn(attn).squeeze(2)
        attn = F.softmax(attn, 1).unsqueeze(1)

        read = (attn * know).sum(2)

        return read


class WriteUnit(nn.Module):
    def __init__(self, dim, self_attention=False, memory_gate=False):
        super().__init__()

        self.concat = linear(dim * 2, dim)

        if self_attention:
            self.attn = linear(dim, 1)
            self.mem = linear(dim, dim)

        if memory_gate:
            self.control = linear(dim, 1)

        self.self_attention = self_attention
        self.memory_gate = memory_gate

    def forward(self, memories, retrieved, controls):
        prev_mem = memories[-1]
        concat = self.concat(torch.cat([retrieved, prev_mem], 1))
        next_mem = concat

        if self.self_attention:
            controls_cat = torch.stack(controls[:-1], 2)
            attn = controls[-1].unsqueeze(2) * controls_cat
            attn = self.attn(attn.permute(0, 2, 1))
            attn = F.softmax(attn, 1).permute(0, 2, 1)

            memories_cat = torch.stack(memories, 2)
            attn_mem = (attn * memories_cat).sum(2)
            next_mem = self.mem(attn_mem) + concat

        if self.memory_gate:
            control = self.control(controls[-1])
            gate = F.sigmoid(control)
            next_mem = gate * prev_mem + (1 - gate) * next_mem

        return next_mem


class MACUnit(nn.Module):
    def __init__(self,
                 dim,
                 max_step=12,
                 ncf=0,
                 self_attention=False,
                 memory_gate=False,
                 dropout=0.15):
        super().__init__()

        self.control = ControlUnit(dim, max_step, ncf=ncf)
        self.read = ReadUnit(dim)
        self.write = WriteUnit(dim, self_attention, memory_gate)

        self.mem_0 = nn.Parameter(torch.zeros(1, dim))
        self.control_0 = nn.Parameter(torch.zeros(1, dim))

        self.dim = dim
        self.max_step = max_step
        self.dropout = dropout

    def get_mask(self, x, dropout):
        mask = torch.empty_like(x).bernoulli_(1 - dropout)
        mask = mask / (1 - dropout)

        return mask

    def forward(self, context, question, knowledge):
        b_size = question.size(0)

        control = self.control_0.expand(b_size, self.dim)
        memory = self.mem_0.expand(b_size, self.dim)

        if self.training:
            control_mask = self.get_mask(control, self.dropout)
            memory_mask = self.get_mask(memory, self.dropout)
            control = control * control_mask
            memory = memory * memory_mask

        controls = [control]
        memories = [memory]

        for i in range(self.max_step):
            control = self.control(i, context, question, control)
            if self.training:
                control = control * control_mask
            controls.append(control)

            read = self.read(memories, knowledge, controls)
            memory = self.write(memories, read, controls)
            if self.training:
                memory = memory * memory_mask
            memories.append(memory)

        return memory


class MACNetwork(nn.Module):
    def __init__(self,
                 vocab,
                 embedding_dim,
                 n_in,
                 nf,
                 ncf=None,
                 with_camera=False,
                 max_step=12,
                 self_attention=False,
                 memory_gate=False,
                 dropout=0.15):
        super().__init__()
        """
        vocab: same as clevr.py
        embedding_dim: same as clevr.py
        n_in: same as clevr.py
        nf: same as clevr.py

        no rnn_dim? it seems to be the same thing as nf
        in this code.
        """

        classes = len(vocab['answer_idx_to_token'])
        n_vocab = len(vocab['question_token_to_idx'])
        embed_hidden = embedding_dim

        if ncf is None:
            if with_camera is True:
                raise Exception("ncf must be != None if use_camera==True")
            ncf = 0
        dim = nf

        self.vocab = vocab
        self.with_camera = with_camera

        self.conv = nn.Sequential(nn.Conv2d(n_in, dim, 3, padding=1),
                                  nn.ELU(),
                                  nn.Conv2d(dim, dim, 3, padding=1),
                                  nn.ELU())

        self.embed = nn.Embedding(n_vocab, embed_hidden)
        self.lstm = nn.LSTM(embed_hidden, dim,
                            batch_first=True,
                            bidirectional=True)
        self.lstm_proj = nn.Linear(dim * 2, dim)

        self.mac = MACUnit(dim=dim,
                           max_step=max_step,
                           self_attention=self_attention,
                           memory_gate=memory_gate,
                           dropout=dropout,
                           ncf=ncf)

        self.classifier = nn.Sequential(linear(dim * 3 + ncf, dim),
                                        nn.ELU(),
                                        linear(dim, classes))

        if with_camera:
            self.camera_mlp = nn.Linear(6, ncf)

        self.max_step = max_step
        self.dim = dim

        self.reset()

    def reset(self):
        self.embed.weight.data.uniform_(0, 1)

        kaiming_uniform_(self.conv[0].weight)
        self.conv[0].bias.data.zero_()
        kaiming_uniform_(self.conv[2].weight)
        self.conv[2].bias.data.zero_()

        kaiming_uniform_(self.classifier[0].weight)

    #def forward(self, image, question, question_len, dropout=0.15):
    def forward(self, z, questions, camera, dropout=0.15):

        # the original code function signature was of
        # the form: image, question, question_len, dropout
        #
        # `image` is just the feature maps
        # `questions` is questions
        # but we need to compute question_len

        N, T = questions.size(0), questions.size(1)
        question_len = torch.LongTensor(N).fill_(T-1)
        for i in range(N):
            for t in range(T - 1):
                if questions[i, t] != 0 and questions[i, t + 1] == 0:
                    question_len[i] = t
                    break

        #b_size = question.size(0)
        b_size = N

        #img = self.conv(image)
        img = self.conv(z)
        img = img.view(b_size, self.dim, -1)

        embed = self.embed(questions)
        # here in our code we need to do enforce_sorted=False
        embed = nn.utils.rnn.pack_padded_sequence(embed, question_len,
                                                  batch_first=True,
                                                  enforce_sorted=False)
        lstm_out, (h, _) = self.lstm(embed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out,
                                                       batch_first=True)
        # lstm_out: of dimension (batch, seq_len, hidden_size)
        # i.e. the output of the lstm at each time step
        lstm_out = self.lstm_proj(lstm_out)

        # h: the hidden state at time t, of dimension
        # (n_layers*n_dir, bs, hidden_dim)
        # where in this case n_dir=2 because bidirectional
        # permute(1,0,2) -> (bs, n_layers*n_dir, hidden_dim)
        # view(bs, -1) -> (bs, n_layers*n_dir, hidden_dim)
        h = h.permute(1, 0, 2).contiguous().view(b_size, -1)

        if self.with_camera:
            camera_embedding = self.camera_mlp(camera)
            h = torch.cat((h, camera_embedding), dim=1)

        # (ctx, question, knowledge)
        memory = self.mac(lstm_out, h, img)

        out = torch.cat([memory, h], 1)
        out = self.classifier(out)

        return out

def get_network(vocab,
                n_in,
                nf,
                ncf,
                with_camera=False,
                embedding_dim=300,
                self_attention=False,
                max_step=12,
                **kwargs):
    print("get_network's kwargs that are ignored:", kwargs)
    probe = MACNetwork(vocab=vocab,
                       n_in=n_in,
                       nf=nf,
                       ncf=ncf,
                       with_camera=with_camera,
                       self_attention=self_attention,
                       embedding_dim=embedding_dim,
                       max_step=max_step)
    return probe

if __name__ == '__main__':
    print("MAC probe")
    probe = get_network(

    )
