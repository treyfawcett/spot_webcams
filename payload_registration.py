# Copyright (c) 2022 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

"""
Example code for using the payload service API
"""
from __future__ import print_function
import argparse
import sys
import logging
import io
import struct
import time

import bosdyn.client
from bosdyn.client.payload import PayloadClient
from bosdyn.client.payload_registration import PayloadRegistrationClient, PayloadRegistrationKeepAlive
import bosdyn.client.util

from bosdyn.client.frame_helpers import *
from bosdyn.client.math_helpers import *

import bosdyn.api.payload_pb2 as payload_protos
import bosdyn.api.robot_id_pb2 as robot_id_protos

LOGGER = logging.getLogger()


def payload_registrant(config):
    """A simple example of using the Boston Dynamics API to communicate payload configs with spot.

    First registers a payload then lists all payloads on robot, including newly registered payload.
    """

    sdk = bosdyn.client.create_standard_sdk('PayloadSpotClient')

    robot = sdk.create_robot(config.hostname)

    # Authenticate robot before being able to use it
    bosdyn.client.util.authenticate(robot)

    # Create a payload registration client
    payload_registration_client = robot.ensure_client(
        PayloadRegistrationClient.default_service_name)

    # Create a payload
    payload = payload_protos.Payload()
    payload.GUID = config.guid
    payload_secret = config.secret
    payload.name = config.payload_name
    payload.description = config.description
    # payload.label_prefix.append("test_payload")
    payload.is_authorized = False
    payload.is_enabled = False
    payload.is_noncompute_payload = True
    payload.version.major_version = 0
    payload.version.minor_version = 0
    payload.version.patch_level = 1
    # note: this field is not required, but highly recommended
    payload.mount_frame_name = payload_protos.MOUNT_FRAME_BODY_PAYLOAD
    
    # Register the payload
    payload_registration_client.register_payload(payload, payload_secret)

    # Create a payload client
    payload_client = robot.ensure_client(PayloadClient.default_service_name)


    # List all payloads
    payloads = payload_client.list_payloads()
    print('\n\nPayload Listing\n' + '-' * 40)
    print(payloads)



def add_payload_arguments(parser):
    parser.add_argument('--payload-name', required=True, type=str, help='Name of payload.')
    parser.add_argument('--guid', required=True, type=str, help='GUID for payload')
    parser.add_argument('--secret', required=True, type=str, help='Secret for payload')
    parser.add_argument('--description', required=False, type=str, help='Human-friendly description of payload')

def main(argv):
    """Command line interface."""
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    options = parser.parse_args(argv)

    payload_registrant(options)

    return True


if __name__ == '__main__':
    if not main(sys.argv[1:]):
        sys.exit(1)
