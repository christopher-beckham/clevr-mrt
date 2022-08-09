import torch
from torch import nn
from .baselines import RnnEncoder
from .layers import ResidualBlock

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        return x.view(x.size(0), -1)

class ClevrProbe(nn.Module):
    """
    How the probe works:

    z -----[MLP]-----> pred
           ^   ^  ^  ^
    [q,c]--|---|--|--|

    where q is the embedded question
    where c is the camera coordinates

    (z may be replaced by h instead)

    """
    def __init__(self,
                 vocab,
                 n_in,
                 nf=None,
                 ncf=None,
                 embedding_dim=300,
                 rnn_dim=256,
                 rnn_num_layers=2,
                 proj_dim=None,
                 n_resblocks=2,
                 with_coords=False,
                 with_camera=False,
                 downsample=False,
                 coord_shape=None,
                 is_3d=False,
                 theta_dim=32,
                 flatten_3d=False,
                 encoder='lstm'):
        """
        nf: number of filters used for the resblocks
        ncf: if not none, the embedding dimension
          used to project the camera coordinates
        embedding_dim:
        rnn_dim: hidden dimension of the RNN
        rnn_num_layers: number of hidden layers in RNN
        flatten_3d: if is_3d is True, then this will
          collapse the depth and feature axes of the
          volume to bring it back into 2d.
        theta_dim: number of hidden units for the
          MLP which maps camera coords to values
          for rotation/translation matrix.
        """
        super(ClevrProbe, self).__init__()

        self.n_out = len(vocab['answer_idx_to_token'])

        self.with_camera = with_camera
        self.flatten_3d = flatten_3d

        self.ncf = 0
        if with_camera:
            if ncf is None:
                self.ncf = 6
            else:
                self.ncf = ncf

        if nf is None:
            self.nf = (rnn_dim+self.ncf+self.ncf) // 2
            print("`nf` is `None` so making it `(rnn_dim+ncf+ncf)/2`...")
            # NOTE: the reason why we do ncf twice is because in the
            # case of using an identity transform on the embedding,
            # we have a part of the camera in the gamma and beta.
        else:
            self.nf = nf

        rnn_kwargs = {
            'token_to_idx': vocab['question_token_to_idx'],
            'wordvec_dim': embedding_dim,
            'rnn_dim': rnn_dim,
            'rnn_num_layers': rnn_num_layers,
            'rnn_dropout': 0,
            'model': encoder
        }
        print("rnn_kwargs:")
        print(rnn_kwargs)
        self.rnn = RnnEncoder(**rnn_kwargs)
        self.rnn_dim = rnn_dim

        # TODO: basically the main MLP
        self.resblocks = nn.ModuleList([])
        for j in range(n_resblocks):
            if j==0:
                self.resblocks.append(
                    ResidualBlock(n_in, self.nf,
                                  with_film=True,
                                  with_coords=with_coords,
                                  coord_shape=coord_shape,
                                  is_3d=is_3d,
                                  downsample=downsample)
                )
            else:
                self.resblocks.append(
                    ResidualBlock(self.nf, self.nf,
                                  with_film=True,
                                  with_coords=with_coords,
                                  coord_shape=coord_shape,
                                  is_3d=is_3d,
                                  downsample=downsample)
                )
        self.embed_convert = nn.ModuleList([])
        if nf is None:
            for j in range(n_resblocks):
                self.embed_convert.append(nn.Identity())
        else:
            for j in range(n_resblocks):
                self.embed_convert.append(
                    nn.Linear(rnn_dim+self.ncf+self.ncf, nf*2)
                )

        self.proj = None
        if proj_dim is not None:
            self.proj = nn.Sequential(
                nn.Conv3d(self.nf, proj_dim, 1) if is_3d else nn.Conv2d(self.nf, proj_dim, 1),
                nn.BatchNorm3d(proj_dim) if is_3d else nn.BatchNorm2d(proj_dim),
                nn.ReLU()
            )

        if is_3d:
            self.pool = nn.AdaptiveAvgPool3d(1)
        else:
            self.pool = nn.AdaptiveAvgPool2d(1)

        if with_camera:
            self.camera_mlp = nn.Linear(6, self.ncf)

        self.flatten = Flatten()
        if proj_dim is None:
            self.fc = nn.Linear(self.nf, self.n_out)
        else:
            self.fc = nn.Linear(proj_dim, self.n_out)

        # This is a silly artifact that needs to be 
        # removed from probe. holo_contrastive_encoder.py
        # depends on this, and nothing else does.
        self.cam_encode_3d = nn.Sequential(
            nn.Linear(6, theta_dim),
            nn.BatchNorm1d(theta_dim),
            nn.ReLU(),
            nn.Linear(theta_dim, 6)
        )

    def forward(self, z, questions, camera):
        """
        Params
        ------

        questions:
        feats: either z or h (depending on model)
        camera: the camera

        Description
        -----------

        (1) Q -> self.rnn -> embed
        (2) C -> self.camera_mlp -> camera_embed
        (3) Create 'total' embedding by doing
            a three-way concatenation:
            [camera_embed, embed, camera_embed]
            (camera_embed == null if camera mode
            is disabled)
        (4) If `nf` is None, then it is assigned
            to be the dimension of the total embedding
            * 2, s.t. no FILM layers are needed and
            gamma = total_embed[0:half],
            beta = total_embed[half::]
            (that way, gamma and beta both have
            camera information, hence concatting
            it on both sides of `embed`)

            If `nf` is not None, then for each
            resblock, make MLP projectors which
            map from the dim of `total_embed` to
            `nf*2`, which is then split in half
            to provide the gamma and beta for
            FILM.
        (5) Run the FILM-modulated resblocks,
            do pool at the end, flatten, and
            feed to classifier MLP.

        """

        if self.flatten_3d:
            z = z.view(-1, z.size(1)*z.size(2), z.size(3), z.size(4))

        # Embed both question and camera.
        embedding = self.rnn(questions)
        if self.with_camera:
            camera_embedding = self.camera_mlp(camera)
            embedding = torch.cat((camera_embedding,
                                   embedding,
                                   camera_embedding), dim=1)

        h = z
        for j in range(len(self.resblocks)):
            #embed2film = self.embed_convert[j](embedding)
            h = self.resblocks[j](h,
                                  self.embed_convert[j](embedding))

        if self.proj is not None:
            h = self.proj(h)
        h = self.pool(h)
        h = self.flatten(h)
        out = self.fc(h)

        return out

def get_network(vocab,
                n_in,
                nf=None,
                ncf=None,
                encoder='lstm',
                embedding_dim=300,
                rnn_dim=256,
                rnn_num_layers=2,
                n_resblocks=2,
                proj_dim=None,
                with_coords=False,
                with_camera=False,
                coord_shape=None,
                downsample=False,
                flatten_3d=False,
                is_3d=False):
    return ClevrProbe(vocab,
                      n_in=n_in,
                      nf=nf,
                      ncf=ncf,
                      encoder=encoder,
                      embedding_dim=embedding_dim,
                      rnn_dim=rnn_dim,
                      rnn_num_layers=rnn_num_layers,
                      n_resblocks=n_resblocks,
                      proj_dim=proj_dim,
                      with_coords=with_coords,
                      with_camera=with_camera,
                      downsample=downsample,
                      coord_shape=coord_shape,
                      flatten_3d=flatten_3d,
                      is_3d=is_3d)

if __name__ == '__main__':

    from .test_baseline import load_vocab

    vocab = load_vocab("/clevr_preprocessed/vocab.json")

    probe = ClevrProbe(
        vocab=vocab
    )

    print(probe)
