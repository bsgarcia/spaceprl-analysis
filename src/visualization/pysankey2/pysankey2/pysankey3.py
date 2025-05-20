from collections import defaultdict, OrderedDict
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import numpy as np
import pandas as pd
import math
from .utils import setColorConf, listRemoveNAN

__all__ = ['Sankey', 'LabelMismatchError']

class SankeyException(Exception):
    pass

class LabelMismatchError(SankeyException):
    """LabelMismatchError is thrown when the provided labels are different from the labels in the dataframe."""
    pass

class Sankey:
    """
    Static Sankey diagram based on matplotlib.
    pySankey3 currently supports 4-layer Sankey diagram, where user can freely set the box position, strip length, etc.
    The returned matplotlib.figure and matplotlib.axes object allows post modification using matplotlib api.
    """
    def __init__(self, dataFrame, layerLabels=None, colorDict=None, colorMode="global", stripColor="grey"):
        """
        Parameters:
        -----------
        dataFrame: pd.DataFrame
            Each row of the dataFrame represents a trans-entity.
        
        layerLabels: dict
            If passing, the provided layerLabels would determine the drawing order of each layer.
            If passing, dict keys must be named corresponding to column names of dataFrame.
                e.g {'layer1':['label1','label2','label4'],'layer2':['label2','label5','label6']}
            If not passing, layerLabels would be extracted from the dataFrame.
        
        colorDict: dict
            There are 2 modes can be passing, see colorMode for details.

        colorMode: str, Can only take option in ["global","layer"].
            If choosing "global", a colorPalette dict that merged the same label in different layers would be taken.
            If choosing "layer", a colorPalette dict that treat the same label in different layers as independent label would be taken.
        
        stripColor: str, specified strip color.
            Default is "grey".
            If choosing "left": The color of strip would be the same as the box on the left.
            Specified colors would be passed into plt.fill_between().
        """
        self.dataFrame = deepcopy(dataFrame)
        self._colnameMaps = self._getColnamesMapping(self.dataFrame)
        self.dataFrame.columns = ['layer%d' % (i + 1) for i in range(dataFrame.shape[1])]

        self._allLabels = self._getAllLabels(self.dataFrame)
        if layerLabels is None:
            self._layerLabels = self._getLayerLabels(self.dataFrame)
        else:
            self._checkLayerLabelsMatchDF(self.dataFrame, layerLabels, self._colnameMaps)
            self._layerLabels = layerLabels

        self.colorMode = colorMode
        _opts = ["global", "layer"]
        if colorMode not in _opts:
            raise ValueError("colorMode options must be one of:{0} ".format(",".join([i for i in _opts])))
        if colorDict is None:
            self._colorDict = self._setColorDict(self._layerLabels, mode=colorMode)
        else:
            self._checkColorMatchLabels(colorDict, mode=colorMode)
            if colorMode == "layer":
                colorDict = self._renameColorDict(colorDict)
            self._colorDict = colorDict

        self._stripColor = stripColor
        self._labelCount = {'layer%d' % (i + 1): self.dataFrame['layer%d' % (i + 1)].value_counts().to_dict() for i in range(dataFrame.shape[1])}
        self._boxPos = self._setboxPos(self.dataFrame, self._layerLabels, boxInterv=0.02)
        self._layerPos = self._setLayerPos(self._layerLabels, boxWidth=2, stripLen=10)
        self._stripWidths = self._setStripWidth(self._layerLabels, self.dataFrame)

    def _getColnamesMapping(self, dataFrame):
        return dict(zip(dataFrame.columns, ['layer%d' % (i + 1) for i in range(dataFrame.shape[1])]))

    def _getAllLabels(self, dataFrame):
        uniqLabels = list(set(dataFrame.unstack().values))
        allLabels = listRemoveNAN(uniqLabels)
        return allLabels

    def _getLayerLabels(self, dataFrame):
        layerLabels = OrderedDict()
        for layer_label in dataFrame.columns:
            layer_labels = list(dataFrame.loc[:, layer_label].unique())
            layer_labels = listRemoveNAN(layer_labels)
            layerLabels[layer_label] = layer_labels
        return layerLabels

    def _checkLayerLabelsMatchDF(self, dataFrame, layerLabels, colnameMaps):
        for old_colname, new_colname in colnameMaps.items():
            if set(dataFrame[old_colname].unique()) != set(layerLabels[new_colname]):
                raise LabelMismatchError(f"Labels in {old_colname} do not match with provided layerLabels")

    def _checkColorMatchLabels(self, colorDict, mode):
        if mode == "global":
            provided_set = set(colorDict.keys())
            df_set = set(self.labels)
        elif mode == "layer":
            provided_set = set([label for layer in colorDict.values() for label in layer.keys()])
            df_set = set([label for layer in self.layerLabels.values() for label in layer])
        if provided_set != df_set:
            msg_provided = "Provided Color Labels:" + ",".join([str(i) for i in provided_set]) + "\n"
            msg_df = "dataFrame Labels:" + ",".join([str(i) for i in df_set]) + "\n"
            raise LabelMismatchError(f'In {mode}, {msg_provided} do not match with {msg_df}')

    def _setColorDict(self, layerLabels, mode):
        if mode == "global":
            ngroups = len(self.labels)
            colorPalette = setColorConf(ngroups=ngroups)
            colorDict = {label: colorPalette[i] for i, label in enumerate(self.labels)}
        elif mode == "layer":
            all_layer_labels = [label for layer_labels in layerLabels.values() for label in layer_labels]
            ngroups = len(all_layer_labels)
            colorPalette = setColorConf(ngroups=ngroups)
            colorDict = defaultdict(dict)
            i = 0
            for layer, layer_labels in layerLabels.items():
                for layer_label in layer_labels:
                    colorDict[layer][layer_label] = colorPalette[i]
                    i += 1
        return colorDict

    def _renameColorDict(self, colorDict):
        for old_name, new_name in self.colnameMaps.items():
            if old_name != new_name:
                colorDict[new_name] = colorDict.pop(old_name)
        return colorDict

    def _setboxPos(self, dataFrame, layerLabels, boxInterv):
        boxPos = OrderedDict()
        for layer, labels in layerLabels.items():
            layerPos = defaultdict(dict)
            for i, label in enumerate(labels):
                labelHeight = dataFrame[dataFrame.loc[:, layer] == label].loc[:, layer].count()
                if i == 0:
                    layerPos[label]['bottom'] = 0
                    layerPos[label]['top'] = labelHeight
                else:
                    prevLabelTop = layerPos[labels[i - 1]]['top']
                    layerPos[label]['bottom'] = prevLabelTop + boxInterv * dataFrame.loc[:, layer].count()
                    layerPos[label]['top'] = layerPos[label]['bottom'] + labelHeight
            boxPos[layer] = layerPos
        return boxPos

    def _setLayerPos(self, layerLabels, boxWidth, stripLen):
        layerPos = defaultdict(dict)
        layerStart = 0
        layerEnd = boxWidth
        for layer in layerLabels.keys():
            layerPos[layer]['layerStart'] = layerStart
            layerPos[layer]['layerEnd'] = layerEnd
            layerStart = layerEnd + stripLen
            layerEnd = layerStart + boxWidth
        return layerPos

    def _setStripWidth(self, layerLabels, dataFrame):
        layers = list(layerLabels.keys())
        stripWidths = defaultdict(lambda: defaultdict(dict))
        for i, layer in enumerate(layers):
            if i == len(layers) - 1:
                break
            leftLayer = layers[i]
            rightLayer = layers[i + 1]
            for leftLabel in layerLabels[leftLayer]:
                for rightLabel in layerLabels[rightLayer]:
                    width = len(dataFrame[(dataFrame.loc[:, leftLayer] == leftLabel) & (dataFrame.loc[:, rightLayer] == rightLabel)])
                    if width > 0:
                        stripWidths[leftLayer][leftLabel][rightLabel] = width
        return stripWidths

    def _setStripPos(self, leftBottom, rightBottom, leftTop, rightTop, kernelSize, stripShrink):
        ys_bottom = np.array(50 * [leftBottom] + 50 * [rightBottom])
        ys_bottom = np.convolve(ys_bottom + stripShrink, (1 / kernelSize) * np.ones(kernelSize), mode='valid')
        ys_bottom = np.convolve(ys_bottom + stripShrink, (1 / kernelSize) * np.ones(kernelSize), mode='valid')

        ys_top = np.array(50 * [leftTop] + 50 * [rightTop])
        ys_top = np.convolve(ys_top - stripShrink, (1 / kernelSize) * np.ones(kernelSize), mode='valid')
        ys_top = np.convolve(ys_top - stripShrink, (1 / kernelSize) * np.ones(kernelSize), mode='valid')
        return ys_bottom, ys_top

    def _plotBox(self, ax, boxPos, layerPos, layerLabels, colorDict, fontSize, fontPos, box_kws, text_kws):
        for layer, labels in layerLabels.items():
            for label in labels:
                labelBot = boxPos[layer][label]['bottom']
                labelTop = boxPos[layer][label]['top']
                layerStart = layerPos[layer]['layerStart']
                layerEnd = layerPos[layer]['layerEnd']

                if self.colorMode == "global":
                    color = colorDict[label]
                elif self.colorMode == "layer":
                    color = colorDict[layer][label]

                ax.fill_between(
                    x=[layerStart, layerEnd],
                    y1=labelBot,
                    y2=labelTop,
                    facecolor=color,
                    alpha=0.9,
                    **box_kws
                )

                distToBoxLeft = fontPos[0]
                distToBoxBottom = fontPos[1]
                ax.text(
                    (layerStart + distToBoxLeft),
                    (labelBot + (labelTop - labelBot) * distToBoxBottom),
                    label + " (" + str(self._labelCount[layer][label]) + ")",
                    {'ha': 'right', 'va': 'center'},
                    fontsize=fontSize,
                    **text_kws
                )

    def _plotStrip(self, ax, dataFrame, layerLabels, boxPos, layerPos, stripWidths, kernelSize, stripShrink, stripColor, strip_kws):
        layers = list(layerLabels.keys())
        for i, layer in enumerate(layers):
            if i == len(layers) - 1:
                break
            leftLayer = layers[i]
            rightLayer = layers[i + 1]
            boxPosProxy = deepcopy(boxPos)
            for leftLabel in layerLabels[leftLayer]:
                for rightLabel in layerLabels[rightLayer]:
                    width = len(dataFrame[(dataFrame.loc[:, leftLayer] == leftLabel) & (dataFrame.loc[:, rightLayer] == rightLabel)])
                    if width > 0:
                        leftBottom = boxPosProxy[leftLayer][leftLabel]['bottom']
                        leftTop = leftBottom + stripWidths[layer][leftLabel][rightLabel]

                        rightBottom = boxPosProxy[rightLayer][rightLabel]['bottom']
                        rightTop = rightBottom + stripWidths[layer][leftLabel][rightLabel]

                        ys_bottom, ys_top = self._setStripPos(leftBottom, rightBottom, leftTop, rightTop, kernelSize=kernelSize, stripShrink=stripShrink)

                        boxPosProxy[leftLayer][leftLabel]['bottom'] = leftTop
                        boxPosProxy[rightLayer][rightLabel]['bottom'] = rightTop

                        x_start = layerPos[leftLayer]['layerEnd']
                        x_end = layerPos[rightLayer]['layerStart']

                        if stripColor == "left":
                            if self.colorMode == "global":
                                ax.fill_between(
                                    np.linspace(x_start, x_end, len(ys_top)), ys_bottom, ys_top, alpha=0.4,
                                    color=self.colorDict[leftLabel],
                                    **strip_kws
                                )
                            elif self.colorMode == "layer":
                                ax.fill_between(
                                    np.linspace(x_start, x_end, len(ys_top)), ys_bottom, ys_top, alpha=0.4,
                                    color=self.colorDict[leftLayer][leftLabel],
                                    **strip_kws
                                )
                        elif stripColor == "right":
                            ax.fill_between(
                                np.linspace(x_start, x_end, len(ys_top)), ys_bottom, ys_top, alpha=0.4,
                                color=self.colorDict[rightLabel],
                                **strip_kws
                            )
                        else:
                            ax.fill_between(
                                np.linspace(x_start, x_end, len(ys_top)), ys_bottom, ys_top, alpha=0.4,
                                color=stripColor,
                                **strip_kws
                            )

    def plot(self, figSize=(10, 10), fontSize=10, fontPos=(-0.15, 0.5), boxInterv=0.02, boxWidth=2, stripLen=10, kernelSize=25, stripShrink=0, box_kws=None, text_kws=None, strip_kws=None, savePath=None):
        if box_kws is None:
            box_kws = {}
        if text_kws is None:
            text_kws = {}
        if strip_kws is None:
            strip_kws = {}

        fig, ax = plt.subplots(figsize=figSize)
        self._plotBox(ax, self.boxPos, self.layerPos, self.layerLabels, self.colorDict, fontSize, fontPos, box_kws, text_kws)
        self._plotStrip(ax, self.dataFrame, self.layerLabels, self.boxPos, self.layerPos, self.stripWidth, kernelSize, stripShrink, self.stripColor, strip_kws)

        if savePath:
            plt.savefig(savePath, dpi=300)
        return fig, ax

    @property
    def colnameMaps(self):
        return self._colnameMaps

    @property
    def labels(self):
        return self._allLabels

    @property
    def layerLabels(self):
        return self._layerLabels

    @property
    def boxPos(self):
        return self._boxPos

    @property
    def layerPos(self):
        return self._layerPos

    @property
    def stripWidth(self):
        return self._stripWidths

    @property
    def colorDict(self):
        return self._colorDict

    @property
    def stripColor(self):
        return self._stripColor