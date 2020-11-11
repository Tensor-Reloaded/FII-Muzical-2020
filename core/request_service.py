from http.server import BaseHTTPRequestHandler, HTTPServer

host = 'localhost'
port = 32008

class requestService(BaseHTTPRequestHandler):
    def set_response(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_GET(self):
        self.set_response()
        self.wfile.write("GET request for {}".format(self.path).encode('utf-8'))
    
    def do_POST(self):
        self.set_response()
        content_length = int(self.headers.get('content-length', 0))
        post_data = self.rfile.read(content_length)
        self.wfile.write("POST request for{}".format(self.path).encode('utf-8'))

HTTPServer((host, port), requestService).serve_forever()
