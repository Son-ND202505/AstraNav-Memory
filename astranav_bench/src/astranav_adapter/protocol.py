# Part of the MuJoCo G1 object-goal navigation release (Apache-2.0, see LICENSE in this folder).
import json
import socket
import struct
import numpy as np

def recv_exact(conn, n: int) -> bytes:
    buf = b''
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError('peer closed the connection')
        buf += chunk
    return buf

def send(conn, header: dict, payload: bytes=b''):
    h = json.dumps(dict(header, payload_len=len(payload))).encode()
    conn.sendall(struct.pack('>I', len(h)) + h + payload)

def receive(conn):
    n = struct.unpack('>I', recv_exact(conn, 4))[0]
    header = json.loads(recv_exact(conn, n).decode())
    return (header, recv_exact(conn, header.get('payload_len', 0)))

class AstraNavClient:

    def __init__(self, host: str='127.0.0.1', port: int=8300, timeout: float=900.0):
        self.sock = socket.create_connection((host, port), timeout=timeout)

    def _call(self, header: dict, payload: bytes=b'') -> dict:
        send(self.sock, header, payload)
        reply, _ = receive(self.sock)
        if 'error' in reply:
            raise RuntimeError(f"model server: {reply['error']}")
        return reply

    def hello(self) -> dict:
        return self._call(dict(op='hello'))

    def reset(self, goal: str, image_length: int=50, color_order: str='rgb') -> dict:
        return self._call(dict(op='reset', goal=goal, image_length=int(image_length), color_order=color_order))

    def set_goal(self, goal: str) -> dict:
        return self._call(dict(op='set_goal', goal=goal))

    def decide(self, rgbs, poses, ref, frontiers, goal: str, score: bool=False) -> dict:
        arr = [np.ascontiguousarray(r, dtype=np.uint8) for r in rgbs]
        h, w = arr[0].shape[:2]
        head = dict(op='decide', n=len(arr), h=int(h), w=int(w), poses=[[float(v) for v in p] for p in poses], ref=[float(v) for v in ref], frontiers=[[float(v) for v in f] for f in frontiers], goal=goal, score=bool(score))
        return self._call(head, b''.join((a.tobytes() for a in arr)))

    def close(self):
        try:
            send(self.sock, dict(op='quit'))
        except Exception:
            pass
        self.sock.close()
