import numpy as np
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg
import time

import helper


Buffer = None
ROIs = None

################################################################
### Eye Position Detector Widget

class EyePositionDetector(QtWidgets.QWidget):
    def __init__(self, parent):
        self.main = parent

        QtWidgets.QWidget.__init__(self, parent, flags=QtCore.Qt.Window)
        self.setWindowTitle('Eye position detector')
        self.setLayout(QtWidgets.QGridLayout())

        self.graphicsWidget = EyePositionDetector.GraphicsWidget(parent=self)
        self.layout().addWidget(self.graphicsWidget, 0, 0)

        #self.frame = None
        #self.frametime = None
        #self.last_frametime = 0.0

    def updateFrame(self):
        #if self.frame is None:
        #    return

        #t = self.frametime.value
        #if t <= self.last_frametime:
        #    return

        idx, frame = Buffer.read('frame')
        idx, times = Buffer.read('time', last=2)
        frametime = times[-1] - times[-2]
        self.graphicsWidget.imageItem.setImage(np.rot90(frame, -1))
        self.graphicsWidget.textItem_fps.setText('FPS: {:.2f}'.format(1/frametime))
        self.graphicsWidget.textItem_time.setText('Time: {:.2f}'.format(times[-1]))

    class GraphicsWidget(pg.GraphicsLayoutWidget):
        def __init__(self, **kwargs):
            pg.GraphicsLayoutWidget.__init__(self, **kwargs)

            ### Set up basics
            self.lineSegROIs = dict()
            self.rectROIs = dict()
            self.rectSubplots = dict()
            self.newMarker = None
            self.currentId = 0

            self.imagePlot = self.addPlot(0, 0, 1, 10)

            ### Set up plot image item
            self.imageItem = pg.ImageItem()
            self.imagePlot.hideAxis('left')
            self.imagePlot.hideAxis('bottom')
            self.imagePlot.setAspectLocked(True)
            #self.imagePlot.vb.setMouseEnabled(x=False, y=False)
            self.imagePlot.addItem(self.imageItem)

            self.imageItem.sigImageChanged.connect(self.updateRectSubplots)
            ### Bind mouse click event for drawing of lines
            self.imagePlot.scene().sigMouseClicked.connect(self.mouseClicked)

            ### Bind context menu call function
            self.imagePlot.vb.raiseContextMenu = self.raiseContextMenu

            ### Add text
            self.textItem_fps = pg.TextItem('', color=(255, 0, 0))
            self.imagePlot.addItem(self.textItem_fps)

            self.textItem_time = pg.TextItem('', color=(255, 0, 0))
            self.textItem_time.setPos(300, 0)
            self.imagePlot.addItem(self.textItem_time)

        def resizeEvent(self, ev):
            pg.GraphicsLayoutWidget.resizeEvent(self, ev)
            self.setEyeRectPlotMaxHeight()

        def setEyeRectPlotMaxHeight(self):
            if not(hasattr(self, 'ci')):
                return
            self.ci.layout.setRowMaximumHeight(1, self.height()//6)

        def raiseContextMenu(self, ev):

            ### Set context menu
            self.menu = QtWidgets.QMenu()

            ## Set new line
            self._menu_act_new = QtWidgets.QAction('New marker line')
            self._menu_act_new.triggered.connect(self.addMarkerLine)
            self.menu.addAction(self._menu_act_new)
            self.menu.popup(QtCore.QPoint(ev.screenPos().x(), ev.screenPos().y()))

        def addMarkerLine(self):
            if self.newMarker is not None:
                return
            self.newMarker = list()

        def mouseClicked(self, ev):
            pos = self.imagePlot.vb.mapSceneToView(ev.scenePos())

            ### First click: start new line
            if self.newMarker is not None and len(self.newMarker) == 0:
                self.newMarker = [[pos.x(), pos.y()]]

            ### Second click: end line and create rectangular ROI + subplot
            elif self.newMarker is not None and len(self.newMarker) == 1:
                ## Set second point of line
                self.newMarker.append([pos.x(), pos.y()])

                ## Draw line
                lineSegROI = EyePositionDetector.Line(self, self.currentId, self.newMarker,
                                                      pen=pg.mkPen(color='FF0000', width=2))
                self.lineSegROIs[self.currentId] = lineSegROI
                self.imagePlot.vb.addItem(self.lineSegROIs[self.currentId])


                rectROI = EyePositionDetector.Rect(self, self.currentId, self.newMarker)
                self.rectROIs[self.currentId] = rectROI
                self.imagePlot.vb.addItem(self.rectROIs[self.currentId])

                ## Add subplot
                self.rectSubplots[self.currentId] = dict()
                sp = self.addPlot(1, self.currentId)
                ii = pg.ImageItem()
                sp.hideAxis('left')
                sp.hideAxis('bottom')
                sp.setAspectLocked(True)
                sp.vb.setMouseEnabled(x=False, y=False)
                sp.addItem(ii)

                self.rectSubplots[self.currentId]['imageitem'] = ii
                self.rectSubplots[self.currentId]['plotitem'] = sp

                self.currentId += 1
                self.newMarker = None

        def updateRectSubplots(self):
            idx, rects = Buffer.read('extracted_rects')

            if rects is None:
                return

            ### Plot rectangualar ROIs
            self.rectSubplots[0]['imageitem'].setImage(np.rot90(rects, -1))


    class Line(pg.LineSegmentROI):
        def __init__(self, parent, id, *args, **kwargs):
            self.parent = parent
            self.id = id
            pg.LineSegmentROI.__init__(self, *args, **kwargs, movable=False, removable=True)


    class Rect(pg.RectROI):

        def __init__(self, parent, id, coords):
            self.parent = parent
            self.id = id
            ## Add rectangle
            lLen = np.linalg.norm(np.array(coords[0]) - np.array(coords[1]))
            pg.RectROI.__init__(self, pos=coords[0], size=lLen * np.array([0.8, 1.3]), movable=False, centered=True)

            self.parent.lineSegROIs[self.id].sigRegionChangeFinished.connect(self.updateRect)
            self.sigRegionChangeFinished.connect(self.updateRect)
            self.updateRect()


        def updateRect(self):
            linePoints = self.parent.lineSegROIs[self.id].listPoints()
            lineCoords = [[linePoints[0].x(), linePoints[0].y()], [linePoints[1].x(), linePoints[1].y()]]
            lineStart = np.array(lineCoords[0])
            lineEnd = np.array(lineCoords[1])
            line = helper.vecNormalize(lineEnd - lineStart)
            lineAngleRad = np.arccos(np.dot(helper.vecNormalize(np.array([-1.0, 0.0])), line))

            if line[1] > 0:
               lineAngleRad = 2*np.pi - lineAngleRad

            self.setPos(lineStart, finish=False)
            self.setAngle(360 * lineAngleRad / (2 * np.pi), finish=False)

            self.translate(-0.5 * self.size().x() * np.array([np.cos(lineAngleRad), np.sin(lineAngleRad)])
                           + 0.5 * self.size().y() * np.array([np.sin(lineAngleRad),-np.cos(lineAngleRad)]),
                           finish=False)

            self.rect = [lineStart, np.array(self.size()), 360 * lineAngleRad / (2 * np.pi)]

            ### Set updates ROI parameters
            ROIs[0] = self.rect


class Plotter(pg.PlotWidget):

    def __init__(self, main):
        pg.PlotWidget.__init__(self)
        self.main = main

        self.le = pg.PlotDataItem(name='Left eye position [deg]', pen=pg.mkPen(color=(0,0,255,255),width=2))
        self.le_pos = list()
        self.addItem(self.le)

        self.re = pg.PlotDataItem(name='Right eye position [deg]', pen=pg.mkPen(color=(255,0,0,255),width=2))
        self.re_pos = list()
        self.addItem(self.re)

        self.legend = pg.LegendItem(offset=(40,0.9))
        #self.plotItem.addLegend()
        self.legend.addItem(self.le, self.le.name())
        self.legend.addItem(self.re, self.re.name())
        self.legend.setParentItem(self.plotItem)

    def updatePlot(self):

        #if not(bool(self.main.eyePositions_left)) or not(bool(self.main.eyePositions_right)):
        #    return
        last = 2000
        _, times = Buffer.read('time', last=last)
        _, le = Buffer.read('le_pos', last=last)
        _, re = Buffer.read('re_pos', last=last)


        self.le.setData(times, le)
        self.re.setData(times, re)


################################################################
### Main window

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, pipein=None, pipeout=None, buffer=None, rois=None):
        self.app = QtWidgets.QApplication([])
        QtWidgets.QMainWindow.__init__(self, flags=QtCore.Qt.Window)
        ### Set values
        self.pipein = pipein
        self.pipeout = pipeout
        global Buffer
        self.buffer = buffer
        Buffer = buffer
        global ROIs
        self.ROIs = rois
        ROIs = rois

        buffer.initialize()

        self.resize(1300, 1000)
        self.move(50, 50)

        ### Set up central widget
        self.wdgt = QtWidgets.QWidget(self)
        self.wdgt.setLayout(QtWidgets.QGridLayout())
        self.setCentralWidget(self.wdgt)

        ### Set up camera widget
        self.cam_wdgt = EyePositionDetector(self)
        self.wdgt.layout().addWidget(self.cam_wdgt, 0, 0)

        ### Set up inputs
        self.input_panel = QtWidgets.QWidget()
        self.wdgt.layout().addWidget(self.input_panel, 0, 1)
        self.input_panel.setLayout(QtWidgets.QGridLayout())
        self.input_panel.layout().addWidget(QtWidgets.QLabel('TEEEST'), 0, 0)

        ### Set up plotter
        self.plot_wdgt = Plotter(self)
        self.wdgt.layout().addWidget(self.plot_wdgt, 1, 0, 1, 2)

        ### Set frame update timer
        self.timer = QtCore.QTimer()
        self.timer.setInterval(30)
        self.timer.timeout.connect(self.cam_wdgt.updateFrame)
        self.timer.timeout.connect(self.plot_wdgt.updatePlot)
        self.timer.start()

        self.show()
        self.app.exec_()

    def closeEvent(self, evt):
        self.pipeout.send([99])
        evt.accept()


