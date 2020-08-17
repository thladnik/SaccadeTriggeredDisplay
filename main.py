"""
SaccadeTriggeredDisplay ./main.py
Copyright (C) 2020 Tim Hladnik

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
"""

import h5py
import tisgrabber as IC
import numpy as np
import time
import wres
import multiprocessing as mp
import ctypes
import cv2
import pyfirmata
from scipy.io import savemat

import algorithm
import gui


################################################################
### Buffer


class BufferDTypes:
    ### Unsigned integers
    uint8 = (ctypes.c_uint8, np.uint8)
    uint16 = (ctypes.c_uint16, np.uint16)
    uint32 = (ctypes.c_uint32, np.uint32)
    uint64 = (ctypes.c_uint64, np.uint64)
    ### Signed integers
    int8 = (ctypes.c_int8, np.int8)
    int16 = (ctypes.c_int16, np.int16)
    int32 = (ctypes.c_int32, np.int32)
    int64 = (ctypes.c_int64, np.int64)
    ### Floating point numbers
    float32 = (ctypes.c_float, np.float32)
    float64 = (ctypes.c_double, np.float64)
    ### Misc types
    dictionary = (dict, )


class RingBuffer:
    """A simple ring buffer model. """

    def __init__(self, buffer_length=1000):
        self.__dict__['_bufferLength'] = buffer_length

        self.__dict__['_attributeList'] = list()
        ### Index that points to the record which is currently being updated
        self.__dict__['_currentIdx'] = manager.Value(ctypes.c_int64, 0)

    def initialize(self):
        for attr_name in self.__dict__['_attributeList']:
            shape = self.__dict__['_shape_{}'.format(attr_name)]
            if shape is None:
                continue

            data = np.frombuffer(self.__dict__['_dbase_{}'.format(attr_name)], self.__dict__['_dtype_{}'.format(attr_name)][1])
            if shape != (1,):
                data = data.reshape((self.length(), *shape))
            self.__dict__['_data_{}'.format(attr_name)] = data

    def next(self):
        self.__dict__['_currentIdx'].value += 1

    def index(self):
        return self.__dict__['_currentIdx'].value

    def length(self):
        return self.__dict__['_bufferLength']

    def read(self, name, last=1, last_idx=None):
        """Read **by consumer**: return last complete record(s) (_currentIdx-1)
        Returns a tuple of (index, record_dataset)
                        or (indices, record_datasets)
        """
        if not(last_idx is None):
            last = self.index()-last_idx

        ### Set index relative to buffer length
        list_idx = (self.index()) % self.length()

        ### One record
        if last == 1:
            idx_start = list_idx-1
            idx_end = None
            idcs = self.index()-1

        ### Multiple record
        elif last > 1:
            if last > self.length():
                raise Exception('Trying to read more records than stored in buffer. '
                                'Attribute \'{}\''.format(name))

            idx_start = list_idx-last
            idx_end = list_idx

            idcs = list(range(self.index()-last, self.index()))

        ### No entry: raise exception
        else:
            raise Exception('Smallest possible record set size is 1')

        if isinstance(name, str):
            return idcs, self._read(name, idx_start, idx_end)
        else:
            return idcs, {n: self._read(n, idx_start, idx_end) for n in name}

    def _createAttribute(self, attr_name, dtype, shape=None):
        self.__dict__['_attributeList'].append(attr_name)
        if shape is None:
            self.__dict__['_data_{}'.format(attr_name)] = manager.list(self.length() * [None])
        else:
            ### *Note to future self*
            # ALWAYS try to use shared arrays instead of managed lists, etc for stuff like this
            # Performance GAIN in the particular example of the Camera process pushing
            # 720x750x3 uint8 images through the buffer is close to _100%_ (DOUBLING of performance)\\ TH 2020-07-16
            self.__dict__['_dbase_{}'.format(attr_name)] = mp.RawArray(dtype[0], int(np.prod([self.length(), *shape])))
            self.__dict__['_data_{}'.format(attr_name)] = None
        self.__dict__['_dtype_{}'.format(attr_name)] = dtype
        self.__dict__['_shape_{}'.format(attr_name)] = shape

    def _read(self, attr_name, idx_start, idx_end):

        ### Return single record
        if idx_end is None:
            return self.__dict__['_data_{}'.format(attr_name)][idx_start]

        ### Return multiple records
        if idx_start >= 0:
            return self.__dict__['_data_{}'.format(attr_name)][idx_start:idx_end]
        else:
            if self.__dict__['_shape_{}'.format(attr_name)] is None:
                return self.__dict__['_data_{}'.format(attr_name)][idx_start:] \
                       + self.__dict__['_data_{}'.format(attr_name)][:idx_end]
            else:
                return np.concatenate((self.__dict__['_data_{}'.format(attr_name)][idx_start:],
                        self.__dict__['_data_{}'.format(attr_name)][:idx_end]), axis=0)

    def __setattr__(self, name, value):
        if not('_data_{}'.format(name) in self.__dict__):
            self._createAttribute(name, *value)
        else:
            # TODO: add checks?
            self.__dict__['_data_{}'.format(name)][self.index() % self.length()] = value

    def __getattr__(self, name):
        """Get current record"""
        try:
            return self.__dict__['_data_{}'.format(name)][(self.index()) % self.length()]
        except:
            ### Fallback to parent is essential for pickling!
            super().__getattribute__(name)


################################################################
### Cameras

class CAM_Virtual:

    exposure = 'exposure'
    gain = 'gain'

    _models = ['Multi_Fish_Eyes_Cam',
                'Single_Fish_Eyes_Cam']

    _formats = {'Multi_Fish_Eyes_Cam' : ['RGB8 (752x480)'],
                'Single_Fish_Eyes_Cam' : ['RGB8 (640x480)']}

    _sampleFile = {'Multi_Fish_Eyes_Cam' : 'Fish_eyes_multiple_fish_30s.avi',
                   'Single_Fish_Eyes_Cam' : 'Fish_eyes_spontaneous_saccades_40s.avi'}

    res_x = 480
    res_y = 640

    def __init__(self):
        self._model = self._models[1]
        self._format = self._formats[self._models[1]]
        self.vid = cv2.VideoCapture(self._sampleFile[self._model])

    @classmethod
    def getModels(cls):
        return cls._models

    def updateProperty(self, propName, value):
        pass

    def getFormats(self):
        return self.__class__._formats[self._model]

    def getImage(self):
        ret, frame = self.vid.read()
        if ret:
            return frame
        else:
            self.vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return self.getImage()

class CAM_TIS:

    exposure = 'exposure'
    gain = 'gain'

    camera_model = 'DMK 23U618 49410244'
    camera_format = 'Y800 (640x480)'
    res_x = 480
    res_y = 640

    def __init__(self):
        self._device = IC.TIS_CAM()

        print('Available devices:', IC.TIS_CAM().GetDevices())
        print('Open device {}'.format(self.camera_model))
        self._device.open(self.camera_model)
        print('Available formats:', self._device.GetVideoFormats())
        self._device.SetVideoFormat(self.camera_format)

        ### Disable auto settings
        self._device.SetPropertySwitch('Gain', 'Auto', 0)
        self._device.enableCameraAutoProperty(4, 0)  # Disable auto exposure (for REAL)

        ### Enable frame acquisition
        self._device.StartLive(0)

    def updateProperty(self, propName, value):
        ### Fetch current exposure
        currExposure = [0.]
        self._device.GetPropertyAbsoluteValue('Exposure', 'Value', currExposure)
        currGain = [0.]
        self._device.GetPropertyAbsoluteValue('Gain', 'Value', currGain)

        if propName == self.exposure and not(np.isclose(value, currExposure[0] * 1000, atol=0.001)):
            print('Set exposure from {} to {} ms'.format(currExposure[0] * 1000, value))
            self._device.SetPropertyAbsoluteValue('Exposure', 'Value', float(value)/1000)

        elif propName == self.gain and not (np.isclose(value, currGain[0], atol=0.001)):
            print('Set gain from {} to {}'.format(currGain[0], value))
            self._device.SetPropertyAbsoluteValue('Gain', 'Value', float(value))


    def getImage(self):
        self._device.SnapImage()
        return self._device.GetImage()


def setLEDs(on):
    ### Reset pins
    for i, p in pins.items():
        p.write(on)

def handleComm():
    if not(pipein.poll()):
        return

    global running, sacc_trigger_mode, sacc_diff_threshold, target_fps, \
        flash_delay, flash_dur, trigger, flash_start, flash, flash_intensity

    msg = pipein.recv()

    if msg[0] == 99:
        running = False
    elif msg[0] == 31:
        print('Set camera exposure to {}ms'.format(msg[1]))
        camera.updateProperty(camera.exposure, msg[1])
    elif msg[0] == 32:
        print('Set camera gain to {}'.format(msg[1]))
        camera.updateProperty(camera.gain, msg[1])
    elif msg[0] == 33:
        print('Set camera target framerate to {}'.format(msg[1]))
        target_fps = msg[1]
    elif msg[0] == 21:
        print('Set flash duration to {}ms'.format(msg[1] * 1000))
        flash_dur = msg[1]
    elif msg[0] == 22:
        print('Set flash delay to {}ms'.format(msg[1] * 1000))
        flash_delay = msg[1]
    elif msg[0] == 23:
        print('Set flash intensity to {:.0f}%'.format(msg[1] * 100))
        flash_intensity = msg[1]
    elif msg[0] == 50:
        toggleRecording()
    elif msg[0] == 61:
        print('Trigger set')
        trigger = True
    elif msg[0] == 62:
        print('Flash started by user')
        flash_start = np.inf
        flash = True
    elif msg[0] == 63:
        print('Flash stopped by user')
        flash_start = np.inf
        flash = False

    else:
        print('Unknown comm code {}'.format(msg))


def toggleRecording():
    global file, filename
    ## Open file
    if file is None:
        filename = time.strftime('%Y-%m-%d-%H-%M-%S')
        global flash_delay, flash_dur, sacc_trigger_mode, sacc_diff_threshold
        file = h5py.File('{}.hdf5'.format(filename), 'w')
        file.attrs['flash_delay'] = flash_delay
        file.attrs['flash_duration'] = flash_dur

        print('Start recordingto file {}'.format(file.filename))
    else:
        print('Stop recording')
        file.close()

        ### Export
        print('Exporting file {}.hdf5 to {}.mat'.format(filename, filename))
        file = h5py.File('{}.hdf5'.format(filename), 'r')
        data = {key:value for key, value in file.attrs.items()}
        data.update({key:value[:] for key, value in file.items()})
        savemat('{}.mat'.format(filename), data)

        file.close()

        file = None
        filename = None

def appendData(key, value):
    global file
    if file is None:
        return

    if not(key in file):
        file.create_dataset(key,
                            shape=(0,1),
                            dtype=type(value),
                            maxshape=(None,1),
                            chunks=(100,1)
                            )
    dset = file[key]

    dset.resize((dset.shape[0]+1, *dset.shape[1:]))
    dset[-1] = value

if __name__ == '__main__':

    ### Set windows timer precision as high as possible
    _, maxres, _ = wres.query_resolution()
    with wres.set_resolution(maxres):

        _, _, curres = wres.query_resolution()
        print('Timing resolution is {}ms'.format(curres/10**4))

        # Manager
        manager = mp.Manager()
        ### Set up mp communication
        # Pipe
        pipein, pipeout = mp.Pipe()
        ROIs = manager.dict()

        ### Set up camera
        #camera = CAM_TIS()
        camera = CAM_Virtual()

        ### Set up buffer
        # Camera
        cbuffer = RingBuffer(buffer_length=3000)
        cbuffer.time = (BufferDTypes.float64, (1,))
        cbuffer.frame = (BufferDTypes.uint8, (camera.res_x, camera.res_y))
        cbuffer.le_pos = (BufferDTypes.float64, (1,))
        cbuffer.re_pos = (BufferDTypes.float64, (1,))
        cbuffer.extracted_rects = (BufferDTypes.dictionary,)
        cbuffer.trigger_sig = (BufferDTypes.uint8, (1,))
        # Display
        dbuffer = RingBuffer(buffer_length=10**7)
        dbuffer.time = (BufferDTypes.float64, (1,))
        dbuffer.flash_level = (BufferDTypes.float64, (1,))

        ### Set up display (Arduino)
        board = pyfirmata.Arduino('COM3')
        pins = dict()
        pins[0] = board.get_pin('d:3:p')

        ### Set up and start GUI process
        guip = mp.Process(target=gui.MainWindow, name='GUI',
                          kwargs=dict(pipein=pipein,
                                      pipeout=pipeout,
                                      cbuffer=cbuffer,
                                      dbuffer=dbuffer,
                                      rois=ROIs))
        guip.start()

        ### Initialize buffers locally
        cbuffer.initialize()
        dbuffer.initialize()

        ### Set up detection algorithm
        detector = algorithm.EyePosDetectRoutine(cbuffer, ROIs, camera.res_x, camera.res_y)

        ### Set flash variables
        flash_intensity = None
        trigger = False
        flash = False
        setLEDs(flash)

        flash_start = np.inf
        flash_dur = None
        flash_delay = None

        ### Set trigger variables:
        sacc_trigger_mode = None
        sacc_diff_threshold = None

        target_display_rate = 1000
        target_fps = None
        running = True

        file = None
        filename = None

        ### Wait for gui to set parameters
        while flash_delay is None \
                or flash_intensity is None \
                or flash_delay is None \
                or target_fps is None:
            handleComm()

        t_start = time.time()
        t = 0
        last_ctime = -np.inf
        last_dtime = -np.inf
        while running:

            ### Handle all communication
            handleComm()

            ### Set time for current interation
            t = time.time() - t_start

            if not(t >= last_dtime + 1/target_display_rate):
                continue

            ################################
            ### Snap, grab and process new frame
            if t >= last_ctime + 1/target_fps:

                frame = camera.getImage()
                cbuffer.frame = frame[:, :, 0]
                cbuffer.time = t

                ### Do calculation
                eyePos = detector._compute(frame)

                ### Trigger
                if not(flash) and trigger:
                    trigger = not(trigger)
                    flash_start = t + flash_delay
                    cbuffer.trigger_sig = 1
                else:
                    cbuffer.trigger_sig = 0

                ### Set new last ctime
                last_ctime = t

                ### Save to file
                appendData('c_time',cbuffer.time)
                appendData('c_le_pos',cbuffer.le_pos)
                appendData('c_re_pos',cbuffer.re_pos)
                appendData('c_trigger_sig',cbuffer.trigger_sig)

                ### Advance buffer
                cbuffer.next()


            ################################
            ### Display

            ### Start flash
            if not(flash) and flash_start <= t:
                print('Flash start')
                ### Set pins
                flash = not(flash)

            ### Reset pins
            if flash and t > (flash_start + flash_dur):
                print('Flash stop')
                flash = not(flash)
                flash_start = np.inf

            setLEDs(float(flash) * flash_intensity)

            ### Flash
            dbuffer.time = t
            dbuffer.flash_level = int(flash) * flash_intensity

            ### Save to file
            appendData('d_time', dbuffer.time)
            appendData('d_flash_level', dbuffer.flash_level)

            ### Set new last dtime
            last_dtime = t

            ### Advance buffer
            dbuffer.next()
