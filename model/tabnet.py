from tqdm import tqdm
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.compose import ColumnTransformer


from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix

from tensorflow_addons.activations import sparsemax
from scipy.special import softmax

import matplotlib.pyplot as plt
import seaborn as sns
def glu(x, n_units=None):
    """Generalized linear unit nonlinear activation."""
    return x[:, :n_units] * tf.nn.sigmoid(x[:, n_units:])


class FeatureBlock(tf.keras.Model):
    """
    Implementation of a FL->BN->GLU block
    """

    def __init__(
            self,
            feature_dim,
            apply_glu=True,
            bn_momentum=0.9,
            fc=None,
            epsilon=1e-5,
    ):
        super(FeatureBlock, self).__init__()
        self.apply_gpu = apply_glu
        self.feature_dim = feature_dim
        units = feature_dim * 2 if apply_glu else feature_dim  # desired dimension gets multiplied by 2
        # because GLU activation halves it

        self.fc = tf.keras.layers.Dense(units, use_bias=False) if fc is None else fc  # shared layers can get re-used
        self.bn = tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=epsilon)

    def call(self, x, training=None):
        x = self.fc(x)  # inputs passes through the FC layer
        x = self.bn(x, training=training)  # FC layer output gets passed through the BN
        if self.apply_gpu:
            return glu(x, self.feature_dim)  # GLU activation applied to BN output
        return x


class FeatureTransformer(tf.keras.Model):
    def __init__(
            self,
            feature_dim,
            fcs=[],
            n_total=4,
            n_shared=2,
            bn_momentum=0.9,
    ):
        super(FeatureTransformer, self).__init__()
        self.n_total, self.n_shared = n_total, n_shared

        kwrgs = {
            "feature_dim": feature_dim,
            "bn_momentum": bn_momentum,
        }

        # build blocks
        self.blocks = []
        for n in range(n_total):
            # some shared blocks
            if fcs and n < len(fcs):
                self.blocks.append(FeatureBlock(**kwrgs, fc=fcs[n]))  # Building shared blocks by providing FC layers
            # build new blocks
            else:
                self.blocks.append(FeatureBlock(**kwrgs))  # Step dependent blocks without the shared FC layers

    def call(self, x, training=None):
        print("here",x.shape)
        # input passes through the first block
        x = self.blocks[0](x, training=training)
        print(x.shape)
        # for the remaining blocks
        for n in range(1, self.n_total):
            # output from previous block gets multiplied by sqrt(0.5) and output of this block gets added
            x = x * tf.sqrt(0.5) + self.blocks[n](x, training=training)
            print(x.shape)
        return x

    @property
    def shared_fcs(self):
        return [self.blocks[i].fc for i in range(self.n_shared)]


class AttentiveTransformer(tf.keras.Model):
    def __init__(self, feature_dim):
        super(AttentiveTransformer, self).__init__()
        self.block = FeatureBlock(
            feature_dim,
            apply_glu=False,
        )

    def call(self, x, prior_scales, training=None):
        x = self.block(x, training=training)
        return sparsemax(x * prior_scales)


class TabNet(tf.keras.Model):
    def __init__(
            self,
            num_features,
            feature_dim,
            output_dim,
            n_step=2,
            n_total=4,
            n_shared=2,
            relaxation_factor=1.5,
            bn_epsilon=1e-5,
            bn_momentum=0.7,
            sparsity_coefficient=1e-5
    ):
        super(TabNet, self).__init__()
        self.output_dim, self.num_features = output_dim, num_features
        self.n_step, self.relaxation_factor = n_step, relaxation_factor
        self.sparsity_coefficient = sparsity_coefficient

        self.bn = tf.keras.layers.BatchNormalization(
            momentum=bn_momentum, epsilon=bn_epsilon
        )

        kargs = {
            "feature_dim": feature_dim + output_dim,
            "n_total": n_total,
            "n_shared": n_shared,
            "bn_momentum": bn_momentum
        }

        # first feature transformer block is built first to get the shared blocks
        self.feature_transforms = [FeatureTransformer(**kargs)]
        self.attentive_transforms = []

        # each step consists out of FT and AT
        for i in range(n_step):
            self.feature_transforms.append(
                FeatureTransformer(**kargs, fcs=self.feature_transforms[0].shared_fcs)
            )
            self.attentive_transforms.append(
                AttentiveTransformer(num_features)
            )

        # Final output layer
        self.head = tf.keras.layers.Dense(6, activation="softmax", use_bias=False)

    def call(self, features, training=None):

        bs = tf.shape(features)[0]  # get batch shape
        out_agg = tf.zeros((bs, self.output_dim))  # empty array with outputs to fill
        prior_scales = tf.ones((bs, self.num_features))  # prior scales initialised as 1s
        importance = tf.zeros([bs, self.num_features])  # importances
        masks = []

        features = self.bn(features, training=training)  # Batch Normalisation
        masked_features = features

        total_entropy = 0.0

        for step_i in range(self.n_step + 1):
            # (masked) features go through the FT
            x = self.feature_transforms[step_i](
                masked_features, training=training
            )

            # first FT is not used to generate output
            if step_i > 0:
                # first half of the FT output goes towards the decision
                out = tf.keras.activations.relu(x[:, : self.output_dim])
                out_agg += out
                scale_agg = tf.reduce_sum(out, axis=1, keepdims=True) / (self.n_step - 1)
                importance += mask_values * scale_agg

            # no need to build the features mask for the last step
            if step_i < self.n_step:
                # second half of the FT output goes as input to the AT
                x_for_mask = x[:, self.output_dim:]
                # apply AT with prior scales
                mask_values = self.attentive_transforms[step_i](
                    x_for_mask, prior_scales, training=training
                )

                # recalculate the prior scales
                prior_scales *= self.relaxation_factor - mask_values

                # multiply the second half of the FT output by the attention mask to enforce sparsity
                masked_features = tf.multiply(mask_values, features)

                # entropy is used to penalize the amount of sparsity in feature selection
                total_entropy += tf.reduce_mean(
                    tf.reduce_sum(
                        tf.multiply(-mask_values, tf.math.log(mask_values + 1e-15)),
                        axis=1,
                    )
                )

                # append mask values for later explainability
                masks.append(tf.expand_dims(tf.expand_dims(mask_values, 0), 3))

        # Per step selection masks
        self.selection_masks = masks

        # Final output
        final_output = self.head(out)

        # Add sparsity loss
        loss = total_entropy / (self.n_step - 1)
        self.add_loss(self.sparsity_coefficient * loss)

        return final_output, importance


x=np.random.random(size=(200,10))
y=np.random.random(size=(200,6))
tab=TabNet(10,100,120)
tab(x)

# tab.predict(x)