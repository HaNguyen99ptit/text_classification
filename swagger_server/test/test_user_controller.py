# coding: utf-8

from __future__ import absolute_import

from flask import json
from six import BytesIO

from swagger_server.test import BaseTestCase


class TestUserController(BaseTestCase):
    """UserController integration test stubs"""

    def test_enter_message(self):
        """Test case for enter_message

        enter message
        """
        query_string = [('enter', 'enter_example')]
        response = self.client.open(
            '/p12821/demo2/1.0.0/checkMessage',
            method='GET',
            query_string=query_string)
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_pass_message(self):
        """Test case for pass_message

        enter message
        """
        query_string = [('enter', 'enter_example')]
        response = self.client.open(
            '/p12821/demo2/1.0.0/checkMessage',
            method='POST',
            content_type='application/json',
            query_string=query_string)
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))


if __name__ == '__main__':
    import unittest
    unittest.main()
