# Copyright (c) 2022 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

"""Register and run the Web Cam Service."""

from concurrent.futures import thread
from copy import deepcopy
import logging
import os
import threading
import time
import cv2
import numpy as np
import yaml

import bosdyn.util
from bosdyn.client.directory_registration import (DirectoryRegistrationClient,
                                                  DirectoryRegistrationKeepAlive)
from bosdyn.client.util import setup_logging
from bosdyn.client.server_util import GrpcServiceRunner
from bosdyn.api import image_pb2
from bosdyn.api import image_service_pb2_grpc
from bosdyn.api import geometry_pb2 as geom
from bosdyn.client.frame_helpers import *
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.graph_nav import GraphNavClient


from bosdyn.client.image_service_helpers import (VisualImageSource, CameraBaseImageServicer,
                                                 CameraInterface, ImageCaptureThread,
                                                 convert_RGB_to_grayscale)
from bosdyn.client.async_tasks import AsyncPeriodicQuery,AsyncTasks

DIRECTORY_NAME = 'web-cam-service'
AUTHORITY = 'robot-web-cam'
SERVICE_TYPE = 'bosdyn.api.ImageService'

_LOGGER = logging.getLogger(__name__)


class AsyncRobotState(AsyncPeriodicQuery):
    def __init__(self, robot_state_client):
        super().__init__("robot_state", robot_state_client, _LOGGER,
                                              period_sec=0.1)

    def _start_query(self):
        return self._client.get_robot_state_async()


class AsyncLocalization(AsyncPeriodicQuery):
    def __init__(self, graph_nav_client):
        super().__init__("localization_status", graph_nav_client, _LOGGER,
                                              period_sec=0.1)

    def _start_query(self):
        return self._client.get_localization_state_async()

class CalibratedWebCam(CameraInterface):
    """Provide access to the latest web cam data using openCV's VideoCapture."""

    def __init__(self, device_name, fps=30, show_debug_information=False, codec="",
                 res_width=-1, res_height=-1,calibration_file=None,cameramtx=None,dist_coeff=None):

        self.show_debug_images = show_debug_information

        #try to set these directly if provided
        self.cameramtx, self.dist_coeff = cameramtx, dist_coeff

        self.extrinisic_params = None

        #but the calibration file always wins
        if not calibration_file:
            calibration_file = device_name+'.yaml'
        if os.path.isfile(calibration_file):
            with open(calibration_file, 'r') as file:
                cal = yaml.safe_load(file)
                assert all(key in cal for key in ('width','height','camera_matrix','distortion_coefficients')), f'malformed calibration file `{calibration_file}`: {cal}'
                res_width = cal['width']
                res_height = cal['height']
                self.cameramtx = np.array(cal['camera_matrix'])
                self.dist_coeff = np.array(cal['distortion_coefficients'])
                self.extrinisic_params = cal['extrinsics']


        # Check if the user is passing an index to a camera port, i.e. "0" to get the first
        # camera in the operating system's enumeration of available devices. The VideoCapture
        # takes either a filepath to the device (as a string), or a index to the device (as an
        # int). Attempt to see if the device name can be cast to an integer before initializing
        # the video capture instance.
        # Someone may pass just the index because on certain operating systems (like Windows),
        # it is difficult to find the path to the device. In contrast, on linux the video
        # devices are all found at /dev/videoXXX locations and it is easier.
        try:
            device_name = int(device_name)
        except ValueError as err:
            # No action if the device cannot be converted. This likely means the device name
            # is some sort of string input -- for example, linux uses the default device ports
            # for cameras as the strings "/dev/video0"
            pass

        # Create the image source name from the device name.
        self.image_source_name = device_name_to_source_name(device_name)

        # OpenCV VideoCapture instance.
        self.capture = cv2.VideoCapture(device_name)
        if not self.capture.isOpened():
            # Unable to open a video capture connection to the specified device.
            err = "Unable to open a cv2.VideoCapture connection to %s" % device_name
            _LOGGER.warning(err)
            raise Exception(err)

        self.capture.set(cv2.CAP_PROP_FPS, fps)
        if res_width > 0 and res_height > 0:
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, res_width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, res_height)
            _LOGGER.info("Capture has resolution: %s x %s" % (self.capture.get(cv2.CAP_PROP_FRAME_WIDTH), self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        # Use the codec input argument to determine the  OpenCV 'FourCC' variable is a byte code specifying
        # the video codec (the compression/decompression software).
        if len(codec) == 4:
            # Converts the codec from a single string to a list of four capitalized characters, then
            # creates a video writer based on those characters and sets this as a property of the
            # main VideoCapture for the camera.
            self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*codec.upper()))
        elif len(codec) > 0:
            # Non-empty codec string, but it isn't the correct four character string we expect.
            raise Exception(
                "The codec argument provided (%s) is the incorrect format. It should be a four character string."
                % codec)

        # Attempt to determine the gain and exposure for the camera.
        self.camera_exposure, self.camera_gain = None, None
        try:
            self.camera_gain = self.capture.get(cv2.CAP_PROP_GAIN)
        except cv2.error as e:
            _LOGGER.warning("Unable to determine camera gain: %s", e)
            self.camera_gain = None
        try:
            self.camera_exposure = self.capture.get(cv2.CAP_PROP_EXPOSURE)
        except cv2.error as e:
            _LOGGER.warning("Unable to determine camera exposure time: %s", e)
            self.camera_exposure = None

        # Determine the dimensions of the image.
        self.rows = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.cols = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))

        self.default_jpeg_quality = 75

        # Determine the pixel format.
        self.supported_pixel_formats = []
        success, image = self.capture.read()
        if success:
            if image.shape[2] == 1:
                self.supported_pixel_formats = [image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8]
            elif image.shape[2] == 3:
                self.supported_pixel_formats = [image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8,
                image_pb2.Image.PIXEL_FORMAT_RGB_U8]
            elif image.shape[2] == 4:
                self.supported_pixel_formats = [image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8,
                image_pb2.Image.PIXEL_FORMAT_RGB_U8, image_pb2.Image.PIXEL_FORMAT_RGBA_U8]

        #using 0 for the alpha scale parameter means the result is the same size as the original image, but contains only valid pixels
        self.do_unidstort = self.cameramtx is not None and self.dist_coeff is not None
        if self.do_unidstort:
            self.newcameramtx, self.roi = cv2.getOptimalNewCameraMatrix(self.cameramtx, self.dist_coeff, (self.rows,self.cols), 0, (self.rows,self.cols))
        else:
            self.newcameramtx = None
            self.roi = None
  

    def blocking_capture(self):
        # Get the image from the video capture.
        capture_time = time.time()
        success, image = self.capture.read()
        if self.show_debug_images:
            _LOGGER.info("Image Capture Result: %s", str(success))
            try:
                cv2.imshow("WebCam Image Capture", image)
                cv2.waitKey(1)
            except Exception:
                _LOGGER.warning("Unable to display the webcam image captured.")
                pass
        if success:
            return image, capture_time
        else:
            raise Exception("Unsuccessful call to cv2.VideoCapture().read()")

    def image_decode(self, image_data, image_proto, image_req):
        pixel_format = image_req.pixel_format
        image_format = image_req.image_format

        
        converted_image_data = image_data
        # Determine the pixel format for the data.
        if converted_image_data.shape[2] == 3:
            # RGB image.
            if pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8:
                converted_image_data = convert_RGB_to_grayscale(
                    cv2.cvtColor(converted_image_data, cv2.COLOR_BGR2RGB))
                image_proto.pixel_format = pixel_format
            else:
                image_proto.pixel_format = image_pb2.Image.PIXEL_FORMAT_RGB_U8
        elif converted_image_data.shape[2] == 1:
            # Greyscale image.
            image_proto.pixel_format = image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8
        elif converted_image_data.shape[2] == 4:
            # RGBA image.
            if pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8:
                converted_image_data = convert_RGB_to_grayscale(
                    cv2.cvtColor(converted_image_data, cv2.COLOR_BGRA2RGB))
                image_proto.pixel_format = pixel_format
            else:
                image_proto.pixel_format = image_pb2.Image.PIXEL_FORMAT_RGBA_U8
        else:
            # The number of pixel channels did not match any of the known formats.
            image_proto.pixel_format = image_pb2.Image.PIXEL_FORMAT_UNKNOWN

        if self.do_unidstort:
            converted_image_data = cv2.undistort(converted_image_data, self.cameramtx, self.dist_coeff, None, self.newcameramtx)


        resize_ratio = image_req.resize_ratio
        quality_percent = image_req.quality_percent

        if resize_ratio < 0 or resize_ratio > 1:
            raise ValueError("Resize ratio %s is out of bounds." % resize_ratio)

        if resize_ratio != 1.0 and resize_ratio != 0:
            image_proto.rows = int(image_proto.rows * resize_ratio)
            image_proto.cols = int(image_proto.cols * resize_ratio)
            converted_image_data = cv2.resize(converted_image_data, (image_proto.cols, image_proto.rows), interpolation = cv2.INTER_AREA)

        # Set the image data.
        if image_format == image_pb2.Image.FORMAT_RAW:
            image_proto.data = np.ndarray.tobytes(converted_image_data)
            image_proto.format = image_pb2.Image.FORMAT_RAW
        elif image_format == image_pb2.Image.FORMAT_JPEG or image_format == image_pb2.Image.FORMAT_UNKNOWN or image_format is None:
            # If the image format is requested as JPEG or if no specific image format is requested, return
            # a JPEG. Since this service is for a webcam, we choose a sane default for the return if the
            # request format is unpopulated.
            quality = self.default_jpeg_quality
            if quality_percent > 0 and quality_percent <= 100:
                # A valid image quality percentage was passed with the image request,
                # so use this value instead of the service's default.
                quality = quality_percent
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
            image_proto.data = cv2.imencode('.jpg', converted_image_data, encode_param)[1].tobytes()
            image_proto.format = image_pb2.Image.FORMAT_JPEG
        else:
            # Unsupported format.
            raise Exception(
                "Image format %s is unsupported." % image_pb2.Image.Format.Name(image_format))

   

def _update_thread(async_task, stop_event):
    while not stop_event.is_set():
        async_task.update()
        time.sleep(0.01)


class CalibratedVisualImageSource(VisualImageSource):
    def __init__(self, image_name, camera_interface, static_transforms, rows=None, cols=None, gain=None, exposure=None, pixel_formats=..., logger=None):
        super().__init__(image_name, camera_interface, rows, cols, gain, exposure, pixel_formats, logger)

        #dict of parent_frame_name to SE3Pose
        self.static_transforms = static_transforms
    
    def create_capture_thread(self):
        """Initialize a background thread to continuously capture images.

        Args:
            capture_period (int): Amount of time (in seconds) between captures to wait
                                  before triggering the next capture. Defaults to
                                  "no wait" between captures.
        """
        self.capture_thread = ImageCaptureThread(self.image_source_name, self.capture_function, 0.0)
        self.capture_thread.start_capturing()


class SpatiallyCorrelatedCameraImageServicer(CameraBaseImageServicer):
    def __init__(self, bosdyn_sdk_robot, service_name, image_sources, logger=None, use_background_capture_thread=True):
        super().__init__(bosdyn_sdk_robot, service_name, image_sources, logger, use_background_capture_thread)

        self._robot_state_client = self.bosdyn_sdk_robot.ensure_client(RobotStateClient.default_service_name)
        self._graph_nav_client = self.bosdyn_sdk_robot.ensure_client(GraphNavClient.default_service_name)

        self._robot_state_task = AsyncRobotState(self._robot_state_client)
        self._localization_task = AsyncLocalization(self._graph_nav_client)

        _task_list = [self._robot_state_task, self._localization_task]
        _async_tasks = AsyncTasks(_task_list)

        self._stop_signal = threading.Event()
        self._async_updates_thread = threading.Thread(target=_update_thread, args=[_async_tasks, self._stop_signal])
        self._async_updates_thread.daemon = True
        self._async_updates_thread.start()


    def GetImage(self, request, context):
        response = super().GetImage(request, context)
        kinematic_state = self._robot_state_task.proto.kinematic_state
        frame_map = kinematic_state.transforms_snapshot.child_to_parent_edge_map

        existing_frames = get_frame_names(kinematic_state.transforms_snapshot)

        for img_resp in response.image_responses:
            #initialize mutable edge map
            edges = {frame:frame_map[frame] for frame in frame_map}
            #add edges for known static transforms
            src_name = img_resp.source.name
            image_transforms = self.image_sources_mapped[src_name].static_transforms
            common_frames = set(image_transforms.keys()).intersection(set(existing_frames))
            self.logger.warning(f'no common frames: {image_transforms.keys()} vs {edges.keys}')
            for frame_name in common_frames:
                edges = add_edge_to_tree(edges, image_transforms[frame_name], frame_name, f'webcam_{src_name}')

            #add edges for localization-based transforms if localization is valid
            if self._localization_task.proto[0].localization:
                localization = self._localization_task.proto.localization
                if localization.waypoint_id != '':
                    edges = add_edge_to_tree(edges, localization.waypoint_tform_body, BODY_FRAME_NAME, f'waypoint_{localization.waypoint_id}')

            img_resp.shot.transforms_snapshot = geom.FrameTreeSnapshot(child_to_parent_edge_map=edges)
            img_resp.shot.frame_name_image_sensor = f'webcam_{src_name}'

        return response

    def __del__(self):
        super().__del__()
        self._stop_signal.set()
        self._async_updates_thread.join()


def device_name_to_source_name(device_name):
    if type(device_name) == int:
        return "video" + str(device_name)
    else:
        return os.path.basename(device_name)




def make_webcam_image_service(bosdyn_sdk_robot, service_name, device_names,
                              show_debug_information=False, logger=None, codec="",
                              res_width=-1, res_height=-1, calibration_file=None):
    image_sources = []
    for device in device_names:
        web_cam = CalibratedWebCam(device, show_debug_information=show_debug_information, codec=codec,
                                    res_width=res_width, res_height=res_height, 
                                    calibration_file=calibration_file)
        img_src = CalibratedVisualImageSource(web_cam.image_source_name, web_cam, web_cam.extrinisic_params,
                                    rows=web_cam.rows, cols=web_cam.cols, gain=web_cam.camera_gain,
                                    exposure=web_cam.camera_exposure,
                                    pixel_formats=web_cam.supported_pixel_formats)
        image_sources.append(img_src)
    return SpatiallyCorrelatedCameraImageServicer(bosdyn_sdk_robot, service_name, image_sources, logger)


def run_service(bosdyn_sdk_robot, port, service_name, device_names, show_debug_information=False,
                logger=None, codec="", res_width=-1, res_height=-1, calibration_file=None):
    # Proto service specific function used to attach a servicer to a server.
    add_servicer_to_server_fn = image_service_pb2_grpc.add_ImageServiceServicer_to_server

    # Instance of the servicer to be run.
    service_servicer = make_webcam_image_service(bosdyn_sdk_robot, service_name, device_names,
                                                 show_debug_information, logger=logger, codec=codec,
                                                 res_width=res_width, res_height=res_height,
                                                 calibration_file=calibration_file)
    return GrpcServiceRunner(service_servicer, add_servicer_to_server_fn, port, logger=logger)


def add_web_cam_arguments(parser):
    parser.add_argument('--device-name', action='append', required=False, default=[],
        help=('Image source to query. If none are passed, it will default to the first available source.'))
    parser.add_argument( '--show-debug-info', action='store_true', required=False,
        help="If passed, openCV will try to display the captured web cam images.")
    parser.add_argument( '--codec', required=False, help="The four character video codec (compression format). For example, " +\
        "this is commonly 'DIVX' on windows or 'MJPG' on linux.", default="")
    parser.add_argument('--res-width', required=False, type=int, default=-1, help="Resolution width (pixels).")
    parser.add_argument('--res-height', required=False, type=int, default=-1, help="Resolution height (pixels).")

    parser.add_argument('--calibration-file', required=False, type=str, default="", 
                        help="Distortion parameters. Applied to all cameras")

if __name__ == '__main__':
    # Define all arguments used by this service.
    import argparse
    parser = argparse.ArgumentParser(allow_abbrev=False)
    bosdyn.client.util.add_base_arguments(parser)
    bosdyn.client.util.add_payload_credentials_arguments(parser)
    bosdyn.client.util.add_service_endpoint_arguments(parser)
    add_web_cam_arguments(parser)
    options = parser.parse_args()

    devices = options.device_name
    if not devices:
        # No sources were provided. Set the default source as index 0 to point to the first
        # available device found by the operating system.
        devices = ["0"]

    # Setup logging to use either INFO level or DEBUG level.
    setup_logging(options.verbose, include_dedup_filter=True)

    # Create and authenticate a bosdyn robot object.
    sdk = bosdyn.client.create_standard_sdk("ImageServiceSDK")
    robot = sdk.create_robot(options.hostname)
    robot.authenticate_from_payload_credentials(*bosdyn.client.util.get_guid_and_secret(options))

    # Create a service runner to start and maintain the service on background thread.
    service_runner = run_service(robot, options.port, DIRECTORY_NAME, devices,
                                 options.show_debug_info, logger=_LOGGER, codec=options.codec,
                                 res_width=options.res_width, res_height=options.res_height, 
                                 calibration_file=options.calibration_file)

    # Use a keep alive to register the service with the robot directory.
    dir_reg_client = robot.ensure_client(DirectoryRegistrationClient.default_service_name)
    keep_alive = DirectoryRegistrationKeepAlive(dir_reg_client, logger=_LOGGER)
    keep_alive.start(DIRECTORY_NAME, SERVICE_TYPE, AUTHORITY, options.host_ip, service_runner.port)

    # Attach the keep alive to the service runner and run until a SIGINT is received.
    with keep_alive:
        service_runner.run_until_interrupt()
