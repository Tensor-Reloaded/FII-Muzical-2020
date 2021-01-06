from http.server import BaseHTTPRequestHandler, HTTPServer
import aspectlib
import json
import random


host = 'localhost'
port = 32008

class requestService(BaseHTTPRequestHandler):
    file_names = []

    def set_response(self, status_code):
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

    @aspectlib.Aspect
    def decorator_manage_POST(*args):
        print("Got called with args: ",args)
        result = yield
        print("Result is: %s" % (result,))

    @decorator_manage_POST
    def manage_POST(self, path, data):
        response = 404
        data_response = {}
        if path == "/upload":
            print(data)
            for key in data:
                if key == "create_music":
                    response = 200
                    name = ''.join(random.choice('ABCDEFGHIJ012345') for i in range(16))
                    self.file_names.append(str(name))
                    data_response= {"file" : str(name)}
        return response, data_response

    @aspectlib.Aspect
    def decorator_manage_GET(*args):
        print("Got called with args: ",args)
        result = yield
        print("Result is: %s" % (result,))

    @decorator_manage_GET
    def manage_GET(self, path):
        if path == "/get_generated_song":
            data = {"resp": self.file_names}
            response = 200
        else:
            data = {}
            response = 404
        return response, data

    def do_GET(self):
        response, data = self.manage_GET(self.path)
        self.set_response(response)
        # self.wfile.write("[GET] request for {}".format(self.path).encode('utf-8'))
        self.wfile.write(json.dumps(data).encode('utf-8'))
    
    def do_POST(self):
        content_length = int(self.headers.get('content-length', 0))
        message = json.loads(self.rfile.read(content_length))
        response, data = self.manage_POST(self.path, message)
        self.set_response(response)
        self.wfile.write(json.dumps(data).encode('utf-8'))

HTTPServer((host, port), requestService).serve_forever()