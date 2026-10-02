# coding=utf-8
import cv2
import numpy as np
import threading
import time
import subprocess
import os
import re
from ctypes import *
from NetSDK.NetSDK import NetClient
from NetSDK.SDK_Callback import fDisConnect, fHaveReConnect, fRealDataCallBackEx2
from NetSDK.SDK_Enum import SDK_RealPlayType, EM_LOGIN_SPAC_CAP_TYPE, EM_REALDATA_FLAG
from NetSDK.SDK_Struct import (C_LLONG, NET_IN_LOGIN_WITH_HIGHLEVEL_SECURITY,
                              NET_OUT_LOGIN_WITH_HIGHLEVEL_SECURITY, LOG_SET_PRINT_INFO)

class DahuaCameraViewer:
    def __init__(self):
        self.loginID = C_LLONG()
        self.playID = C_LLONG()
        self.sdk = NetClient()
        self.m_DisConnectCallBack = fDisConnect(self.on_disconnect)
        self.m_ReConnectCallBack = fHaveReConnect(self.on_reconnect)
        self.m_RealDataCallBack = fRealDataCallBackEx2(self.on_frame_data)
        self.raw_buffer = bytearray()
        self.frame_queue = []
        self.frame_lock = threading.Lock()
        self.is_connected = False
        self.is_playing = False
        self.current_frame = None
        self.ffmpeg_process = None
        self.frame_width = None
        self.frame_height = None
        self.resolution_event = threading.Event()
        self.sdk.InitEx(self.m_DisConnectCallBack)
        self.sdk.SetAutoReconnect(self.m_ReConnectCallBack)

    def connect(self, ip, port, username, password):
        print(f"Connecting to {ip}:{port}...")
        if self.loginID:
            print("Already connected!")
            return True
        stuInParam = NET_IN_LOGIN_WITH_HIGHLEVEL_SECURITY()
        stuInParam.dwSize = sizeof(NET_IN_LOGIN_WITH_HIGHLEVEL_SECURITY)
        stuInParam.szIP = ip.encode()
        stuInParam.nPort = port
        stuInParam.szUserName = username.encode()
        stuInParam.szPassword = password.encode()
        stuInParam.emSpecCap = EM_LOGIN_SPAC_CAP_TYPE.TCP
        stuOutParam = NET_OUT_LOGIN_WITH_HIGHLEVEL_SECURITY()
        stuOutParam.dwSize = sizeof(NET_OUT_LOGIN_WITH_HIGHLEVEL_SECURITY)
        self.loginID, device_info, error_msg = self.sdk.LoginWithHighLevelSecurity(stuInParam, stuOutParam)
        if self.loginID != 0:
            self.is_connected = True
            print(f"Connected successfully! Channels: {device_info.nChanNum}")
            return True
        else:
            print(f"Connection failed: {error_msg}")
            return False

    def start_stream(self, channel=0, stream_type=0):
        if not self.is_connected:
            print("Not connected to camera!")
            return False
        if self.playID:
            print("Stream already started!")
            return True
        print(f"Starting stream on channel {channel}...")
        play_type = SDK_RealPlayType.Realplay if stream_type == 0 else SDK_RealPlayType.Realplay_1
        self.playID = self.sdk.RealPlayEx(self.loginID, channel, 0, play_type)
        if self.playID != 0:
            self.sdk.SetRealDataCallBackEx2(self.playID, self.m_RealDataCallBack, None, EM_REALDATA_FLAG.RAW_DATA)
            self.is_playing = True
            print("Stream started successfully!")
            self.setup_decoder()
            return True
        else:
            print(f"Failed to start stream: {self.sdk.GetLastErrorMessage()}")
            return False

    def setup_decoder(self, codec_hint='h264'):
        ffmpeg_path = 'services\\sdk\\dist\\ffmpeg.exe'  #for windows, you might need to specify full path like 'C:\\ffmpeg\\bin\\ffmpeg.exe'
        try:
            cmd = [
                'ffmpeg', '-hwaccel', 'none', '-fflags', 'nobuffer', '-flags', 'low_delay',
                '-probesize', '32', '-analyzeduration', '0', '-f', codec_hint,
                '-i', 'pipe:0', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                '-an', '-loglevel', 'info', 'pipe:1'
            ]
            self.current_codec = codec_hint
            print(f"Setting up decoder for codec: {codec_hint}")
            self.ffmpeg_process = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            self.resolution_parser_thread = threading.Thread(target=self._parse_ffmpeg_resolution, daemon=True)
            self.resolution_parser_thread.start()
            self.decoder_thread = threading.Thread(target=self.decode_frames, daemon=True)
            self.decoder_thread.start()
            print("Decoder setup completed. Waiting for resolution...")
        except FileNotFoundError:
            print("❌ FATAL ERROR: 'ffmpeg' command not found.")
            print("Please install FFmpeg and ensure it is in your system's PATH.")
            self.is_playing = False
        except Exception as e:
            print(f"Failed to setup decoder: {e}")
            self.is_playing = False

    def _parse_ffmpeg_resolution(self):
        res_pattern = re.compile(r'Stream #.*: Video:.*, (\d{2,})x(\d{2,})')
        resolution_found = False
        
        # Vòng lặp này phải chạy liên tục để dọn dẹp stderr
        # cho đến khi tiến trình ffmpeg kết thúc.
        for line in iter(self.ffmpeg_process.stderr.readline, b''):
            if not self.is_playing:
                break # Thoát nếu stream đã chủ động dừng
                
            line_str = line.decode('utf-8', errors='ignore')

            # Chỉ tìm độ phân giải nếu chúng ta chưa tìm thấy
            if not resolution_found:
                match = res_pattern.search(line_str)
                if match:
                    self.frame_width = int(match.group(1))
                    self.frame_height = int(match.group(2))
                    print(f"✅ Resolution detected: {self.frame_width}x{self.frame_height}")
                    self.resolution_event.set() # Báo cho luồng decode_frames bắt đầu
                    resolution_found = True
            
            # (Tùy chọn) Bạn có thể in log của ffmpeg ra đây để gỡ lỗi nếu muốn
            # import sys
            # print(f"[ffmpeg_stderr]: {line_str.strip()}", file=sys.stderr)

        print("FFmpeg stderr reader thread finished.")
        
        # Xử lý trường hợp ffmpeg thoát trước khi tìm thấy độ phân giải
        if not resolution_found and not self.resolution_event.is_set():
            print("⚠️ FFmpeg process exited before resolution could be determined.")
            # Vẫn phải set event để luồng decode_frames không bị treo vĩnh viễn
            self.resolution_event.set()
            
    def decode_frames(self):
        print("⏳ Decoder thread started, waiting for resolution signal...")
        if not self.resolution_event.wait(timeout=10):
            print("❌ Timed out waiting for resolution. Decoder thread will exit.")
            self.is_playing = False
            return
        bytes_per_frame = self.frame_width * self.frame_height * 3
        while self.is_playing and self.ffmpeg_process:
            try:
                frame_data = self.ffmpeg_process.stdout.read(bytes_per_frame)
                if len(frame_data) == bytes_per_frame:
                    frame = np.frombuffer(frame_data, dtype=np.uint8)
                    frame = frame.reshape((self.frame_height, self.frame_width, 3))
                    with self.frame_lock:
                        self.current_frame = frame.copy()
                elif len(frame_data) == 0 and self.ffmpeg_process.poll() is not None:
                    print("FFmpeg process has terminated. Stopping decoder.")
                    break
                else:
                    time.sleep(0.001)
            except Exception as e:
                print(f"Decode error: {e}")
                break
        print("Decoder thread finished.")

    def stop_stream(self):
        if self.playID:
            self.sdk.StopRealPlayEx(self.playID)
            self.playID = 0
        self.is_playing = False
        if self.ffmpeg_process:
            try:
                if self.ffmpeg_process.stdin: self.ffmpeg_process.stdin.close()
                self.ffmpeg_process.terminate()
                self.ffmpeg_process.wait(timeout=2)
            except Exception as e:
                print(f"Error closing ffmpeg process: {e}")
            self.ffmpeg_process = None
        print("Stream stopped")

    def disconnect(self):
        self.stop_stream()
        if self.loginID:
            self.sdk.Logout(self.loginID)
            self.loginID = 0
            self.is_connected = False
            print("Disconnected")

    def cleanup(self):
        self.disconnect()
        self.sdk.Cleanup()
        print("Cleanup completed")

    def on_disconnect(self, lLoginID, pchDVRIP, nDVRPort, dwUser):
        print("Camera disconnected!")
        self.is_connected = False

    def on_reconnect(self, lLoginID, pchDVRIP, nDVRPort, dwUser):
        print("Camera reconnected!")
        self.is_connected = True

    def on_frame_data(self, lRealHandle, dwDataType, pBuffer, dwBufSize, param, dwUser):
        if lRealHandle == self.playID and dwDataType == 0:

            data = cast(pBuffer, POINTER(c_ubyte * dwBufSize)).contents
            raw_bytes = bytes(data)
            if self.ffmpeg_process and self.ffmpeg_process.stdin and not self.ffmpeg_process.stdin.closed:
                try:
                    self.ffmpeg_process.stdin.write(raw_bytes)
                    self.ffmpeg_process.stdin.flush()
                except (BrokenPipeError, OSError):
                    pass
                except Exception as e:
                    print(f"Error writing to ffmpeg stdin: {e}")

    def get_frame(self):
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None