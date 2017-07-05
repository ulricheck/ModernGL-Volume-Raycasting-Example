# Demo for Volume Raycasting using ModernGL
# Author: Ulrich Eck https://github.com/ulricheck
# Code for OpenGL Camera Control borrowed from: https://github.com/pyqtgraph/pyqtgraph/blob/develop/pyqtgraph/opengl/GLViewWidget.py
# Code for Volume Raycasting adapted from: https://github.com/toolchainX/Volume_Rendering_Using_GLSL

import os
import struct
import time

import numpy as np
# @Todo XXX dependency to pyrr should be removed
from pyrr import Matrix44

from PyQt5 import QtCore, QtGui, QtOpenGL, QtWidgets
from OpenGL.GL import glEnable, glDisable, glCullFace, GL_CULL_FACE, GL_FRONT, GL_BACK
from OpenGL.GL import glBindFramebuffer, GL_FRAMEBUFFER

try:
    import ModernGL
except ImportError as e:
    print(""" !!! Please make sure you have successfully installed ModernGL (pip install ModernGL) and PyQt5 !!!""")
    raise e

def load_transferfunction(filename):
    with open(filename,'rb') as fid:
        data = np.fromfile(fid, count=-1,dtype=np.ubyte)
    return data


def load_raw(filename, volsize):
    """ inspired by mhd_utils from github"""
    dim = 3
    element_channels = 1
    np_type = np.ubyte

    arr = list(volsize)
    volume = np.prod(arr[0:dim - 1])

    shape = (arr[dim - 1], volume, element_channels)
    with open(filename,'rb') as fid:
        data = np.fromfile(fid, count=np.prod(shape),dtype = np_type)
    data.shape = shape

    arr.reverse()
    data = data.reshape(arr)
    
    return data

def load_shader(filename):
    with open(os.path.join("glsl", filename), "r") as file:
        shadercode = file.read();
    return shadercode

class QGLControllerWidget(QtOpenGL.QGLWidget):
    def __init__(self, volume_data, volsize, transferfunction, parent=None):
        fmt = QtOpenGL.QGLFormat()
        fmt.setVersion(4, 1)
        fmt.setProfile(QtOpenGL.QGLFormat.CoreProfile)
        fmt.setSampleBuffers(True)
        
        super(QGLControllerWidget, self).__init__(fmt, None)

        self.setFocusPolicy(QtCore.Qt.ClickFocus)

        self.volume_data = volume_data
        self.volume_size = volsize
        self.tff_data = transferfunction

        self.volume_texture = None
        self.tff_texture = None
        self.prog_rc = None
        self.prog_eep = None

        self.vao_eep = None
        self.vao_rc = None
        self.fbo = None

        # Samplig rate controls the distance of the raycasting samples
        self.sampling_rate = 0.5

        self.camera_center = QtGui.QVector3D(0.5, 0.5, 0.5)  ## will always appear at the center of the volume
        self.camera_distance = 3.0          ## distance of camera from center
        self.camera_fov =  60               ## horizontal field of view in degrees
        self.camera_elevation =  30         ## camera's angle of elevation in degrees
        self.camera_azimuth = 45            ## camera's azimuthal angle in degrees 
                                            ## (rotation around z-axis 0 points along x-axis)

        self.noRepeatKeys = [QtCore.Qt.Key_Right, QtCore.Qt.Key_Left, QtCore.Qt.Key_Up, QtCore.Qt.Key_Down, QtCore.Qt.Key_PageUp, QtCore.Qt.Key_PageDown]
        self.keysPressed = {}
        self.keyTimer = QtCore.QTimer()
        self.keyTimer.timeout.connect(self.evalKeyState)

    def initializeGL(self):
        """ Initializes the OpenGL Context and pipeline, and creates + uploads textures for the volume and transfer function """
        self.ctx = ModernGL.create_context()
        self.resizeGL(self.width(), self.height())

        self.volume_texture = self.ctx.texture3d(self.volume_size, 1, self.volume_data.tobytes())
        self.volume_texture.repeat_x = True
        self.volume_texture.repeat_y = True
        # @Todo: ModernGL this raises an error - probably missing wrapper
        #self.volume_texture.repeat_z = True
        self.volume_texture.filter = ModernGL.LINEAR

        self.tff_texture = self.ctx.texture((256,1), 4, self.tff_data.tobytes())
        self.tff_texture.repeat_x = True
        self.tff_texture.repeat_y = True
        self.tff_texture.filter = ModernGL.NEAREST


        self.unf_screensize = None
        self.unf_stepsize = None
        self.unf_transferfunc = None

        self.color_texture = None
        self.depth_texture = None

        self.volume_texture.use(0)
        self.tff_texture.use(1)

        # These are the vertices that make up our cube bounding volume. Every row specifies
        # one corner of our unit cube
        self.vbo_vertex = self.ctx.buffer(struct.pack(
            '24f',
            0.0, 0.0, 0.0,
            0.0, 0.0, 1.0,
            0.0, 1.0, 0.0,
            0.0, 1.0, 1.0,
            1.0, 0.0, 0.0,
            1.0, 0.0, 1.0,
            1.0, 1.0, 0.0,
            1.0, 1.0, 1.0
        ))

        # This is the index buffer for our bounding geometry. Every row specifies a triangle
        # by three indices of our vbo_index vertex buffer
        self.vbo_veridx = self.ctx.buffer(struct.pack(
            '36I',
            1,5,7,
            7,3,1,
            0,2,6,
            6,4,0,
            0,1,3,
            3,2,0,
            7,5,4,
            4,6,7,
            2,3,7,
            7,6,2,
            1,0,4,
            4,5,1
        ))

        self.reload_shaders()

    def reload_shaders(self):
        print("Reloading shaders..")
        self.prog_eep = self.ctx.program([
            self.ctx.vertex_shader(load_shader("exitpoints.vert")),
            self.ctx.fragment_shader(load_shader("exitpoints.frag")),
        ])

        self.prog_rc = self.ctx.program([
            self.ctx.vertex_shader(load_shader("raycasting.vert")),
            self.ctx.fragment_shader(load_shader("raycasting.frag")),
        ])
        
        # re-attach vertex array objects
        vbo_attributes = ['VerPos',]
        vbo_format = ModernGL.detect_format(self.prog_eep, vbo_attributes)
        self.vao_eep = self.ctx.vertex_array(self.prog_eep, [(self.vbo_vertex, vbo_format, vbo_attributes)], self.vbo_veridx)

        vbo_format = ModernGL.detect_format(self.prog_rc, vbo_attributes)
        self.vao_rc = self.ctx.vertex_array(self.prog_rc, [(self.vbo_vertex, vbo_format, vbo_attributes)], self.vbo_veridx)

        # update handles for the uniforms
        self.unf_screensize = self.prog_rc.uniforms["ScreenSize"]
        self.unf_stepsize = self.prog_rc.uniforms["StepSize"]
        self.unf_transferfunc = self.prog_rc.uniforms["TransferFunc"]
        self.unf_exitpoints = self.prog_rc.uniforms["ExitPoints"]
        self.unf_volumetex = self.prog_rc.uniforms["VolumeTex"]

    def setup_camera(self, prog, near=0.1, far=1000., fovy=45.):
        ratio = self.width() / self.height()
        mvp = Matrix44.perspective_projection(fovy, ratio, near, far)
        mvp *= self.viewMatrix()
        prog.uniforms['Mvp'].write(mvp.astype('float32').tobytes())

    def draw_box_eep(self):
        # decide which face to render (@Todo: ModernGL needs a way to handle this)
        glEnable(GL_CULL_FACE);
        glCullFace(GL_FRONT);
        self.vao_eep.render()
        glDisable(GL_CULL_FACE);

    def draw_box_rc(self):
        # decide which face to render (@Todo: ModernGL needs a way to handle this)
        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);
        self.vao_rc.render()
        glDisable(GL_CULL_FACE);

    def paintGL(self):
        render_start = time.perf_counter()

        w, h = self.width()*self.devicePixelRatio(), self.height()*self.devicePixelRatio()
        self.ctx.enable(ModernGL.DEPTH_TEST)
        self.ctx.viewport = (0, 0, w, h)

        self.ctx.clear(0.9, 0.9, 0.9)

        # render Entry/Exit-Point Texture into FBO
        if self.color_texture is None or self.depth_texture is None:
            # needs to be reset on window resize
            self.color_texture = self.ctx.texture((w,h), 4, None)
            self.depth_texture = self.ctx.depth_texture((w,h), None)
            self.fbo = self.ctx.framebuffer(self.color_texture, self.depth_texture)
        
        # set up and draw to the offscreen buffer for the exit points
        self.fbo.clear()
        self.fbo.use()
        self.setup_camera(self.prog_eep)
        self.draw_box_eep()

        if hasattr(ModernGL, "default_framebuffer"):
            ModernGL.default_framebuffer.use()
        else:
            # stop using the fbo (@Todo: ModernGL needs a way to handle this)
            glBindFramebuffer(GL_FRAMEBUFFER, 0);

        # set up to bind the color texture to the third texture location
        self.color_texture.use(2)

        # clear the buffer and set up the uniforms
        self.ctx.clear(0.9, 0.9, 0.9)
        self.unf_screensize.value = (w,h) 
        self.unf_stepsize.value = self.sampling_rate / max(self.volume_size) # step size is relative to the number of voxels
        self.unf_volumetex.value = 0    # the volume texture is bound to the first location
        self.unf_transferfunc.value = 1 # the transfer function texture is bound to the second location
        self.unf_exitpoints.value = 2   # the exit points texture is bound to the third location
        self.setup_camera(self.prog_rc)

        # do a front face pass to perform the actual raycasting
        self.draw_box_rc()

        self.ctx.finish()
        self.update()

        # FPS Counter
        dt = time.perf_counter() - render_start
        if hasattr(self, "last_frame_counter"):
            self.last_frame_counter += 1
            self.last_frame_time_acc += dt
            if self.last_frame_counter == 10:
                avg_frame_time = self.last_frame_time_acc / 10
                print("%.3f ms, %.1f FPS" % (avg_frame_time*1000, 1./(avg_frame_time)))
                self.last_frame_time_acc = 0
                self.last_frame_counter = 1
        else:
            self.last_frame_time_acc = dt
            self.last_frame_counter = 1

    def resizeGL(self, width, height):
        self.color_texture = None
        self.depth_texture = None


    def viewMatrix(self):
        tr = QtGui.QMatrix4x4()
        tr.translate( 0.0, 0.0, -self.camera_distance)
        tr.rotate(self.camera_elevation, 1, 0, 0)
        tr.rotate(self.camera_azimuth, 0, 0, -1)
        center = self.camera_center
        tr.translate(-center.x(), -center.y(), -center.z())
        return np.array(tr.copyDataTo()).reshape((4,4)).transpose()

    def setCameraPosition(self, pos=None, distance=None, elevation=None, azimuth=None):
        if distance is not None:
            self.camera_distance = distance
        if elevation is not None:
            self.camera_elevation = elevation
        if azimuth is not None:
            self.camera_azimuth = azimuth
        self.update()
        

    def cameraPosition(self):
        """Return current position of camera based on center, dist, elevation, and azimuth"""
        center = self.camera_center
        dist = self.camera_distance
        elev = self.camera_elevation * np.pi/180.
        azim = self.camera_azimuth * np.pi/180.
        
        pos = QtGui.QVector3D(
            center.x() + dist * np.cos(elev) * np.cos(azim),
            center.y() + dist * np.cos(elev) * np.sin(azim),
            center.z() + dist * np.sin(elev)
        )
        
        return pos

    def orbit(self, azim, elev):
        """Orbits the camera around the center position. *azim* and *elev* are given in degrees."""
        self.camera_azimuth += azim
        #self.opts['elevation'] += elev
        self.camera_elevation = np.clip(self.camera_elevation + elev, -90, 90)
        self.update()
        
    def pan(self, dx, dy, dz, relative=False):
        """
        Moves the center (look-at) position while holding the camera in place. 
        
        If relative=True, then the coordinates are interpreted such that x
        if in the global xy plane and points to the right side of the view, y is
        in the global xy plane and orthogonal to x, and z points in the global z
        direction. Distances are scaled roughly such that a value of 1.0 moves
        by one pixel on screen.
        
        """
        if not relative:
            self.camera_center += QtGui.QVector3D(dx, dy, dz)
        else:
            cPos = self.cameraPosition()
            cVec = self.camera_center - cPos
            dist = cVec.length()  ## distance from camera to center
            xDist = dist * 2. * np.tan(0.5 * self.camera_fov * np.pi / 180.)  ## approx. width of view at distance of center point
            xScale = xDist / self.width()
            zVec = QtGui.QVector3D(0,0,1)
            xVec = QtGui.QVector3D.crossProduct(zVec, cVec).normalized()
            yVec = QtGui.QVector3D.crossProduct(xVec, zVec).normalized()
            self.camera_center = self.camera_center + xVec * xScale * dx + yVec * xScale * dy + zVec * xScale * dz
        self.update()

    def mousePressEvent(self, ev):
        self.mousePos = ev.pos()
        
    def mouseMoveEvent(self, ev):
        diff = ev.pos() - self.mousePos
        self.mousePos = ev.pos()
        
        if ev.buttons() == QtCore.Qt.LeftButton:
            self.orbit(-diff.x(), -diff.y())
        elif ev.buttons() == QtCore.Qt.MidButton:
            if (ev.modifiers() & QtCore.Qt.ControlModifier):
                self.pan(diff.x(), 0, diff.y(), relative=True)
            else:
                self.pan(diff.x(), diff.y(), 0, relative=True)
        
    def mouseReleaseEvent(self, ev):
        pass
        
    def wheelEvent(self, ev):
        delta = 0
        delta = ev.angleDelta().x()
        if delta == 0:
            delta = ev.angleDelta().y()
        if (ev.modifiers() & QtCore.Qt.ControlModifier):
            self.camera_fov *= 0.999**delta
        else:
            self.camera_distance *= 0.999**delta
        self.update()

    def keyPressEvent(self, ev):
        if ev.key() == QtCore.Qt.Key_Escape:
            QtCore.QCoreApplication.quit()
        elif ev.key() == QtCore.Qt.Key_F5:
            self.reload_shaders()
        if ev.key() in self.noRepeatKeys:
            ev.accept()
            if ev.isAutoRepeat():
                return
            self.keysPressed[ev.key()] = 1
            self.evalKeyState()
      
    def keyReleaseEvent(self, ev):
        if ev.key() in self.noRepeatKeys:
            ev.accept()
            if ev.isAutoRepeat():
                return
            try:
                del self.keysPressed[ev.key()]
            except:
                self.keysPressed = {}
            self.evalKeyState()
        
    def evalKeyState(self):
        speed = 2.0
        if len(self.keysPressed) > 0:
            for key in self.keysPressed:
                if key == QtCore.Qt.Key_Right:
                    self.orbit(azim=-speed, elev=0)
                elif key == QtCore.Qt.Key_Left:
                    self.orbit(azim=speed, elev=0)
                elif key == QtCore.Qt.Key_Up:
                    self.orbit(azim=0, elev=-speed)
                elif key == QtCore.Qt.Key_Down:
                    self.orbit(azim=0, elev=speed)

                self.keyTimer.start(16)
        else:
            self.keyTimer.stop()

def main():

    # assumes unsigned byte datatype and volume dimensions of 256x256x225
    volsize = (256, 256, 225)
    volume = load_raw(os.path.join("data", "head256.raw"), volsize)
    tff = load_transferfunction(os.path.join("data", "tff.dat"))

    app = QtWidgets.QApplication([])
    window = QGLControllerWidget(volume, volsize, tff)
    window.move(QtWidgets.QDesktopWidget().rect().center() - window.rect().center())
    window.show()
    app.exec_()

if __name__ == '__main__':
    main()