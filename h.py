import sys, math
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
#https://www.learnopencv.com/homography-examples-using-opencv-python-c/

class GraphicsScene(QGraphicsScene):
    #Create signal exporting QPointF position.
    SceneClicked = pyqtSignal(QPointF)

    def __init__(self, parent=None):
        QGraphicsScene.__init__(self, parent)

        self.setSceneRect(-100, -100, 200, 200)
        self.opt = ""

    def setOption(self, opt):
        self.opt = opt

    def mousePressEvent(self, event):
        # #Emit the signal
        self.SceneClicked.emit(QPointF(event.scenePos()))

class ImageViewer(QGraphicsView):
    ImageClicked = pyqtSignal(QPoint)

    def __init__(self, parent):
        super(ImageViewer, self).__init__(parent)
        self._zoom = 0
        self._empty = True
        self._scene = GraphicsScene()
        self._Image = QGraphicsPixmapItem()
        self._scene.addItem(self._Image)
        self.setScene(self._scene)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QBrush(QColor(30, 30, 30)))
        self.setFrameShape(QFrame.NoFrame)

        #Connect the signal emitted by the GraphicsScene mousePressEvent to relay event
        self._scene.SceneClicked.connect(self.SceneClicked)

    def hasImage(self):
        return not self._empty

    def fitInView(self, scale=True):
        rect = QRectF(self._Image.pixmap().rect())
        if not rect.isNull():
            self.setSceneRect(rect)
            if self.hasImage():
                unity = self.transform().mapRect(QRectF(0, 0, 1, 1))
                self.scale(1 / unity.width(), 1 / unity.height())
                viewrect = self.viewport().rect()
                scenerect = self.transform().mapRect(rect)
                factor = min(viewrect.width() / scenerect.width(),
                             viewrect.height() / scenerect.height())
                self.scale(factor, factor)
            self._zoom = 0

    def setImage(self, pixmap=None):
        self._zoom = 0
        if pixmap and not pixmap.isNull():
            self._empty = False
            self.setDragMode(QGraphicsView.NoDrag)
            self._Image.setPixmap(pixmap)
        else:
            self._empty = True
            self.setDragMode(QGraphicsView.NoDrag)
            self._Image.setPixmap(QPixmap())
        self.fitInView()

    def wheelEvent(self, event):
        if self.hasImage():
            if event.angleDelta().y() > 0:
                factor = 1.1
                self._zoom += 1
            else:
                factor = 0.9
                self._zoom -= 1

            if self._zoom > 0:
                self.scale(factor, factor)
            elif self._zoom == 0:
                self.fitInView()
            else:
                self._zoom = 0

    def toggleDragMode(self):
        if self.dragMode() == QGraphicsView.ScrollHandDrag:
            self.setDragMode(QGraphicsView.NoDrag)
        else:
            self.setDragMode(QGraphicsView.ScrollHandDrag)

    def mousePressEvent(self, event):
        super(ImageViewer, self).mousePressEvent(event)

    def SceneClicked(self, pos):
        # Pass local (scene) coordinates to ImageClicked()
        if self._Image.isUnderMouse():
            self.ImageClicked.emit(pos.toPoint())


class Window(QWidget):
    def __init__(self):
        super(Window, self).__init__()
        self.viewer = ImageViewer(self)
        self.surface = ImageViewer(self)
        # 'Load image' button
        self.btnLoad = QToolButton(self)
        self.btnLoad.setText('Load image')
        self.btnLoad.clicked.connect(self.loadImage)

        self.cboSurfaces = QComboBox()
        for s in ("issia", "ncaacourt", "ncaafield", "netball", "hockey", "rugby", "tennis", "pool"):
            self.cboSurfaces.addItem(s)
        self.cboSurfaces.setCurrentText("pool")
        self.cboSurfaces.currentIndexChanged.connect(self.loadSurface)

        # Button to change from drag/pan to getting pixel info
        self.btnAddCorrespondances = QToolButton(self)
        self.btnAddCorrespondances.setText('Add Correspondance')
        self.btnAddCorrespondances.clicked.connect(self.addCorrespondances)
        self.btnComputeHomograhy = QToolButton(self)
        self.btnComputeHomograhy.setText('Compute Homograhy')
        self.btnComputeHomograhy.clicked.connect(self.computeHomograhy)
        self.editImageCoordsInfo = QLineEdit(self)
        self.editImageCoordsInfo.setReadOnly(True)
        # self.editModelCoords = QLineEdit(self)
        # self.editModelCoords.setReadOnly(False)
        # self.editModelCoords.returnPressed.connect(self.addCorrespondances)
        self.listCorrespondances = QListWidget()
        self.viewer.ImageClicked.connect(self.ImageClicked)
        self.surface.ImageClicked.connect(self.SurfaceClicked)
        self._mylastImagePairs = {0,0}
        self.addingCorrespondancesEnabled = False
        # Arrange layout
        VBlayout = QVBoxLayout(self)
        HB_images_layout = QHBoxLayout()

        HB_images_layout.addWidget(self.viewer)
        HB_images_layout.addWidget(self.surface)
        VBlayout.addLayout(HB_images_layout)
        # VBlayout.addWidget(self.listCorrespondances)
        # self.setSpaceAction=QAction("Set Space", self, shortcut=Qt.Key_Space, triggered=self.setSpace)
        # self.addAction(self.setSpaceAction)
        # VBlayout.addWidget(QPushButton("Space", self, clicked=self.setSpaceAction.triggered))


        HBlayout = QHBoxLayout()
        HBlayout.setAlignment(Qt.AlignLeft)
        HBlayout.addWidget(self.btnLoad)
        HBlayout.addWidget(self.btnAddCorrespondances)
        HBlayout.addWidget(self.editImageCoordsInfo)
        # HBlayout.addWidget(self.editModelCoords)
        HBlayout.addWidget(self.cboSurfaces)
        HBlayout.addWidget(self.btnComputeHomograhy)
        VBlayout.addLayout(HBlayout)

        #Initial data:
        #x-image, y-image, x-model, y-model
        self._my_correspondances = []
        self._my_correspondances.append({'cx':460, 'cy':223, 'rx': 0, 'ry': 0})
        self._my_correspondances.append({'cx':1245, 'cy':454, 'rx': 25, 'ry': 25})
        self._my_correspondances.append({'cx':1152, 'cy':125, 'rx': 25, 'ry': -25})
        self._my_correspondances.append({'cx':541, 'cy':101, 'rx': 0, 'ry': -25})
        print(self._my_correspondances)

        for c in self._my_correspondances:
            self.listCorrespondances.addItem("{0}, {1} ,{2}, {3}".format(c['cx'], c['cy'], c['rx'], c['ry']))

    def resetControls(self):

        #Abort corresponances
        self._mylastImagePairs = {0,0}
        self._mylastSurfacePairs = {0,0}
        self.viewer.setBackgroundBrush(QBrush(QColor(30, 30, 30)))
        self.surface.setBackgroundBrush(QBrush(QColor(30, 30, 30)))
        self.btnAddCorrespondances.setStyleSheet("background-color: None")
        self.addingCorrespondancesEnabled = False
        self.viewer.setDragMode(QGraphicsView.NoDrag)
        self.surface.setDragMode(QGraphicsView.NoDrag)


    def keyPressEvent(self, event):

        if not event.isAutoRepeat():
            if event.key() == Qt.Key_Escape:
                #Abort corresponances
                self.resetControls()
                return

            if self.viewer._empty or self.surface._empty:
                return

            if event.key() == Qt.Key_Space:
                self.viewer.toggleDragMode()
                self.surface.toggleDragMode()

    def keyReleaseEvent(self, event):
        if not event.isAutoRepeat():
            if event.key() == Qt.Key_Space:
                self.viewer.toggleDragMode()
                self.surface.toggleDragMode()

    def loadSurface(self):
        print("Loading surface:", self.cboSurfaces.currentText())
        self.surface.setImage(QPixmap("./Surfaces/{:s}.png".format(self.cboSurfaces.currentText())))

        #Draw point
        # pen = QPen(Qt.red)
        # brush = QBrush(Qt.yellow)
        # for c in self._my_correspondances:
        #     self.surface._scene.addEllipse(c['cx']-3, c['cy']-3, 6, 6, pen, brush)

    def loadImage(self):
        self.viewer.setImage(QPixmap("./hq3.png"))

        #Draw point
        # pen = QPen(Qt.red)
        # brush = QBrush(Qt.yellow)
        # for c in self._my_correspondances:
        #     self.viewer._scene.addEllipse(c['cx']-3, c['cy']-3, 6, 6, pen, brush)

    def pixInfo(self):
        # self.viewer.toggleDragMode()
        if self.addingCorrespondancesEnabled:
            self.viewer.setBackgroundBrush(QBrush(QColor(30, 100, 30)))
            self.surface.setBackgroundBrush(QBrush(QColor(30, 30, 30)))

    def ImageClicked(self, pos):
        print("Image")
        if self.viewer.dragMode()  == QGraphicsView.NoDrag and self.addingCorrespondancesEnabled == True:
            # self.editImageCoordsInfo.setText('%d, %d' % (pos.x(), pos.y()))
            print(pos.x(), pos.y())
            #Draw point
            pen = QPen(Qt.red)
            brush = QBrush(Qt.yellow)
            self.viewer._scene.addEllipse(pos.x()-3, pos.y()-3, 6, 6, pen, brush)
            self.viewer.toggleDragMode()
            self._mylastImagePairs = {pos.x(), pos.y()}
            # self.editModelCoords.setStyleSheet("background-color: rgb(0, 255, 0);")
            self.viewer.setBackgroundBrush(QBrush(QColor(30, 30, 30)))
            self.surface.setBackgroundBrush(QBrush(QColor(30, 100, 30)))
            print("_mylastImagePairs:", self._mylastImagePairs)

    def SurfaceClicked(self, pos):
        print("Surface")
        if self.surface.dragMode()  == QGraphicsView.NoDrag and self.addingCorrespondancesEnabled == True:
            # self.editImageCoordsInfo.setText('%d, %d' % (pos.x(), pos.y()))
            print(pos.x(), pos.y())
            #Draw point
            pen = QPen(Qt.red)
            brush = QBrush(Qt.yellow)
            self.surface._scene.addEllipse(pos.x()-3, pos.y()-3, 6, 6, pen, brush)
            self.surface.toggleDragMode()
            self._mylastSurfacePairs = {pos.x(), pos.y()}
            # self.editModelCoords.setStyleSheet("background-color: rgb(0, 255, 0);")
            self.viewer.setBackgroundBrush(QBrush(QColor(30, 30, 30)))
            self.surface.setBackgroundBrush(QBrush(QColor(30, 30, 30)))
            print("_mylastSurfacePairs:", self._mylastSurfacePairs)

            self.editImageCoordsInfo.setText("{0}, {1} ,{2}, {3}".format(list(self._mylastImagePairs)[1], list(self._mylastImagePairs)[0], list(self._mylastSurfacePairs)[1], list(self._mylastSurfacePairs)[0]))

            #Save corresponances
            self.resetControls()

    def addCorrespondances(self):
        #1. Highlight image space.
        if not self.addingCorrespondancesEnabled:
            self.addingCorrespondancesEnabled = True
            self.btnAddCorrespondances.setStyleSheet("background-color: green")

    def _draw(self, img, corners, imgpts):
        corner = tuple(corners[0].ravel())
        img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
        img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
        img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
        return img

    def cv(self):
        import numpy as np
        import cv2
        import glob
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6*7,3), np.float32)
        objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.
        images = glob.glob('*.jpg')
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (7,6), None)
            # If found, add object points, image points (after refining them)
            if ret == True:
                # objpoints.append(objp)
                corners2=cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
                # imgpoints.append(corners)
                # Draw and display the corners
                cv2.drawChessboardCorners(img, (7,6), corners2, ret)
                # cv2.imshow('img', img)

                # height, width, channel = img.shape
                # bytesPerLine = 3 * width
                # cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
                # qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
                # self.viewer.setImage(QPixmap(qImg))

            # ret, mtx, dist, _, __ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
            height, width, channel = img.shape
            print("Height:", height, "Width:", width)
            # print("Camera Matrix")
            # print(mtx)

            size = img.shape
            focal_length = size[1]
            center = (size[1]/2, size[0]/2)
            print(center)
            mtx = np.array([[focal_length, 0, center[0]],
                                     [0, focal_length, center[1]],
                                     [0, 0, 1]], dtype = "double"
                                     )

            print ("Camera Matrix Manual:\n {0}".format(mtx))

            # return
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            objp = np.zeros((6*7,3), np.float32)
            objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
            axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

            # print("Distortion Matrix")
            # print(dist)
            dist = np.zeros((4,1)) # Assuming no lens distortion

            # Find the rotation and translation vectors.
            # ret,rvecs, tvecs, inliers = cv2.solvePnP(objp, corners2, mtx, dist)

            print("corners2")
            print(corners2)
            print(corners2.__class__.__name__)

            #NB so we CAN use manually estimated camera instrinsic matrix values and nil (zeros) distortion matrix values
            # as inputs to the solve PnP algo.
            (success, rotation_vector, translation_vector) = cv2.solvePnP(objp, corners2, mtx, dist)
            #NB, that will give us the rotation and translation vectors we need to project points.
            # We can then use the projectPoints function with rotation, translation, camera instrinsics and distortion matrices to
            # get 3d camera coordates.

            print("rotation_vector\n", rotation_vector)
            print("translation_vector\n", translation_vector)


            # project 3D points to image plane
            imgpts, jac = cv2.projectPoints(axis, rotation_vector, translation_vector, mtx, dist)
            print("image points")
            print(imgpts)

            img = self._draw(img,corners2,imgpts)

            height, width, channel = img.shape
            bytesPerLine = 3 * width
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
            qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
            self.surface.setImage(QPixmap(qImg))



    def computeHomograhy(self):
        '''
        pts_src and pts_dst are numpy arrays of points
        in source and destination images. We need at least
        4 corresponding points.
        '''
        # pts_src = np.array([[460, 223], [1245, 454], [1152, 125],[541, 101]])
        # pts_dst = np.array([[250, 250], [500, 500], [500, 0],[250, 0]])

        pts_src = np.array([[832, 889], [155, 1394], [3046, 887],[3695, 1412]], dtype='float32')
        pts_src = np.array([(832, 889), (155, 1394), (3046, 887),(3695, 1412)], dtype='float32')
        pts_dst = np.array([[10, 10], [10, 260], [510, 10],[510, 260]], dtype='float32')
        pts_dst = np.array([ (10, 10, 0), (10, 260, 0), (510, 10, 0), (510, 260, 0) ], dtype='float32')

        # Compute 2D homography
        h, status = cv2.findHomography(pts_src, pts_dst)

        '''
        The calculated homography can be used to warp
        the source image to destination. Size is the
        size (width,height) of im_dst
        '''
        im_src = cv2.imread("./hq3.png")

        # Warp source image to destination based on homography
        im_out = cv2.warpPerspective(im_src, h, (520,270))

        # Display images in QT
        height, width, channel = im_out.shape
        bytesPerLine = 3 * width
        cv2.cvtColor(im_out, cv2.COLOR_BGR2RGB, im_out)
        qImg = QImage(im_out.data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.surface.setImage(QPixmap(qImg))

        self.viewer.setBackgroundBrush(QBrush(QColor(30, 30, 30)))
        self.surface.setBackgroundBrush(QBrush(QColor(30, 30, 30)))

        # Render pool boundaries.
        cv2.line(im_src, ( int(pts_src[0][0]), int(pts_src[0][1])), ( int(pts_src[1][0]), int(pts_src[1][1])), (255,255,0), 2)
        cv2.line(im_src, ( int(pts_src[2][0]), int(pts_src[2][1])), ( int(pts_src[1][0]), int(pts_src[1][1])), (255,0,255), 2)
        cv2.line(im_src, ( int(pts_src[2][0]), int(pts_src[2][1])), ( int(pts_src[3][0]), int(pts_src[3][1])), (0,255,0), 2)
        cv2.line(im_src, ( int(pts_src[0][0]), int(pts_src[0][1])), ( int(pts_src[3][0]), int(pts_src[3][1])), (0,255,255), 2)


        # Compute inverse of 2D homography
        val, H = cv2.invert(h)

        # Estimate naive camera intrinsics (camera matrix)
        size = im_src.shape
        focal_length = size[1]
        center = (size[1]/2, size[0]/2)
        camera_matrix = np.array(
                                 [[focal_length, 0, center[0]],
                                 [0, focal_length, center[1]],
                                 [0, 0, 1]], dtype = "float32")

        print ("Camera Matrix :\n {0}".format(camera_matrix))

        # ...assuming no lens distortion
        distortion_matrix = np.zeros((4,1))

        #NB Generate square calibration corresponances using existing homography.
        # The problem is how to learn the camera pose, which we need to estimate a 3D camera coordinate system.
        # We need a) the camera instrinics, such as focal length and distortion, and b) camera extrinsics, rotation and translation.
        # Normally we would like to use a checkerboard for calibrating the camera, and deriving the camera instrinics.
        # OpenCV has lots of established medthods for doing this - such as cv2.CameraCalibration().
        # That's not very convenient for ad-hoc camera calibration, but there are some plausible workarounds.
        # Since we can easily estimate a 2D homography using points corresponding between the image and world coordinate systems,
        # that should give us a relible planar scene context to solve the camera extrinsics.
        # First we estimate naively the focal point (see camera_matrix above), and we assume no distortion(!) *more on that.
        # So, here we use a grid of equi-spaced points in world coordinates, and use the inverse homography to reliably retrieve the
        # (x,y) loation of the corresponding points in camera coordinates.  The perspective embedded in the grid-based camera coordinates should
        # approximate the same points learned from a checkerboard, and we use the cv2.solvePnP() function the give us the extrinsics: rotation and translation.
        # At this point, if we create a grid across the entire image space, the sover does it's best least-suqares approximation of the
        # extrinsics, bu we find when we project points back to the image space that the perspective model fails at a rate proportional to the
        # distance from the center of the image.  I suspect this is because of our previous assumption of zero camera distortion.
        # A hacky workaround is to use a local set of ground truth coordinates, and use the cv2.solvePnP() function
        # to derive compute quasi-extrinsic parameters.  In other words, let w(x,y,z) be the 3D coordinates in world space that we
        # want to project into 2D camera space c(x,y).  We take w(x,y,0), and build a 1m x 1m grid surrounding that point (ensuring we don't breach the real camera bounds).
        # For each point in the 1x1m grid we use the inverse homography transform and directly compute the 2D image coordinates, from which we can compute local
        # camera extrinsics.  Obviously this is a hack, and it would be nice to have more accurate global extrinsics parameters, but we would otherwise
        # need some way of computing the camera distortion accurately.

        _global_calibration = False

        if _global_calibration:
            # Manually set world point calibration for global extrinsics.
            model_width = 50
            model_height = 25
            model_offset_x = 1
            model_offset_y = 1
            world_points = np.zeros((model_width*model_height,3), np.float32)
            world_points[:,:2] = np.mgrid[model_offset_x:model_width+model_offset_x,model_offset_y:model_height+model_offset_y].T.reshape(-1,2)*10 #convert to meters (scale is 1:10)
            camera_points = np.zeros((model_width*model_height,2), np.float32)
        else:
            # Manually set world point calibration for local extrinsics.

            # Top left of pool
            world_points = np.float32([[10,10,0], [60,10,0], [10,60,0],  [60,60,0]])
            camera_points = np.zeros((2*2,2), np.float32)

            # Bottom right of pool space
            # world_points = np.float32([[510,250,0], [500,250,0], [510,240,0],  [500,240,0]])
            # camera_points = np.zeros((2*2,2), np.float32)


        # Estimate image point coordinates on the ground plane for world point calibration markers.
        i = 0
        for world_point in world_points:
            #Remove z-axis (assumed to be zero)
            point = np.array([[[world_point[0], world_point[1]]]], dtype='float32')
            #Get image coordinates for world coordinate point (x,y).
            camera_points[i] = cv2.perspectiveTransform(point, H)
            i = i + 1


        # Solve rotation and translation matrices
        (_, rotation_vector, translation_vector) = cv2.solvePnP(world_points, camera_points, camera_matrix, distortion_matrix)


        # Top left of pool space
        axis = np.float32([[60,10,0], [10,60,0], [10,10,-50]]).reshape(-1,3)
        # Bottom right of pool space
        # axis = np.float32([[500,250,0], [510,240,0], [510,250,-50]]).reshape(-1,3)

        # Render reference point annotation.
        (imgpts, jacobian) = cv2.projectPoints(axis, rotation_vector, translation_vector, camera_matrix, distortion_matrix)
        im_src = self._draw(im_src,camera_points,imgpts)


        if True:
            # This code demonstrates the problem with ignoring instrinsic camera distortion.
            # It should take the inverse homography image points (reliable), and project in the z-plane
            # using the camera extrinsics (rotation/translation) solved above using cv2.solvePnP().
            # If the 3D image coordinate system is accurate, the vertical yellow lines should be
            # generally consistent with normal perspective.
            # By switching between different calibration routines (above), we can see the effects of
            # poor or non-local calibration.
            # For instance, if we use a local calibration grid above, ~10m around the origin (10,10), then the
            # yellow verticals around that region are good, but the non-local distortion corrupts the 3D perspective.
            # Using a global calibration grid (i.e. a grid over the entire model space), then only the points near the
            # center of the image are in proper perspective.

            model_width = 50
            model_height = 25
            model_offset_x = 1
            model_offset_y = 1
            world_points = np.zeros((model_width*model_height,3), np.float32)
            world_points[:,:2] = np.mgrid[model_offset_x:model_width+model_offset_x,model_offset_y:model_height+model_offset_y].T.reshape(-1,2)*10 #convert to meters (scale is 1:10)
            camera_points = np.zeros((model_width*model_height,2), np.float32)

            for world_point in world_points:
                # ground_point = np.array([[[world_point[0], world_point[1], 0]]], dtype='float32')
                # (ground_point, jacobian) = cv2.projectPoints(ground_point, rotation_vector, translation_vector, camera_matrix, distortion_matrix)
                ground_point = np.array([[[world_point[0], world_point[1]]]], dtype='float32')
                ground_point = cv2.perspectiveTransform(ground_point, H)
                ref_point = np.array([[[world_point[0], world_point[1], -20]]], dtype='float32')
                (ref_point, jacobian) = cv2.projectPoints(ref_point, rotation_vector, translation_vector, camera_matrix, distortion_matrix)
                # Render vertical
                im_src = cv2.line(im_src, tuple(ground_point.ravel()), tuple(ref_point.ravel()), (0,255,255), 2)

            if not cv2.imwrite('output.png',im_src):
                print("Writing failed")


        # Display images
        height, width, channel = im_src.shape
        bytesPerLine = 3 * width
        # Convert to RGB for QImage.
        cv2.cvtColor(im_src, cv2.COLOR_BGR2RGB, im_src)
        qImg = QImage(im_src.data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.viewer.setImage(QPixmap(qImg))
        # self.viewer._scene.addEllipse(c['cx']-3, c['cy']-3, 6, 6, pen, brush)



if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    window = Window()
    window.setGeometry(500, 300, 800, 600)
    window.show()
    window.loadImage()
    window.loadSurface()
    sys.exit(app.exec_())
