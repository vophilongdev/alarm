# coding=utf-8

import os
import sys
import ctypes
from ctypes import *
import cv2
import numpy as np
from services.sdk.dist.HCNetSDK import *
from services.sdk.dist.PlayCtrl import *
import threading
import gc
from collections import deque


class HIKCameraViewer:
    """
    Hikvision Camera Viewer - Cross Platform (Windows & Linux)
    """
    
    def __init__(self, dll_path='services/sdk/dist', max_frame_buffer=3):
        """
        Args:
            dll_path: Đường dẫn đến thư mục chứa SDK libraries
            max_frame_buffer: Số frame tối đa giữ trong buffer
        """
        self.dll_path = dll_path
        self.platform = sys.platform
        self.Objdll = None
        self.Playctrldll = None
        self.PlayCtrl_Port = c_long(-1)
        self.FuncDecCB = None
        self.funcRealDataCallBack = None
        
        self.lUserId = -1
        self.lRealPlayHandle = -1
        
        self.frame_buffer = deque(maxlen=max_frame_buffer)
        self.frame_lock = threading.Lock()
        self.frame_count = 0
        self.is_connected = False
        self.is_streaming = False
        
        self.gc_counter = 0
        self.gc_interval = 100
        
        self._load_dll()
        self._init_sdk()
    
    def _load_dll(self):
        """Load SDK libraries theo platform"""
        try:
            if 'win' in self.platform:
                # ... (giữ nguyên code Windows) ...
                pass

            elif 'linux' in self.platform:
                # Linux: sử dụng .so
                hcnet_lib_name = 'libhcnetsdk.so'
                playctrl_lib_name = 'libPlayCtrl.so'
                
                # Lưu lại thư mục hiện tại
                cwd = os.getcwd()
                
                try:
                    # 1. Chuyển thư mục làm việc vào thư mục SDK
                    print(f"Changing working directory to: {self.dll_path}")
                    os.chdir(self.dll_path)
                    
                    # 2. Load dependencies (Nếu cần thiết, nhưng thường chdir là đủ)
                    # self._load_linux_dependencies() 

                    # 3. Load thư viện chính bằng tên file (không cần đường dẫn tuyệt đối nữa)
                    # Lưu ý: dùng "./" để chỉ định load tại thư mục hiện tại
                    self.Objdll = ctypes.cdll.LoadLibrary(f"./{hcnet_lib_name}")
                    self.Playctrldll = ctypes.cdll.LoadLibrary(f"./{playctrl_lib_name}")
                    
                    print(f"✓ SDK libraries loaded successfully via os.chdir trick")

                except Exception as e:
                    raise e
                finally:
                    # 4. Quan trọng: Chuyển lại về thư mục gốc ban đầu
                    os.chdir(cwd)
            
            else:
                raise Exception(f"Unsupported platform: {self.platform}")

        except Exception as e:
            raise Exception(f"Failed to load SDK libraries: {e}")
    
    def _load_linux_dependencies(self):
        """Load các thư viện phụ thuộc trên Linux"""
        try:
            # Danh sách dependencies quan trọng
            # Lưu ý: Thứ tự rất quan trọng, libAudioRender nên được load sớm
            dep_libs = [
                'libanalyzedata.so',
                'libAudioRender.so',  # <--- File đang bị lỗi
                'libHCCore.so',
                'libhpr.so',
                'libHCPreview.so',
                'libSuperRender.so'   # Thường đi kèm với AudioRender
            ]
            
            for lib in dep_libs:
                lib_path = os.path.join(self.dll_path, lib)
                if os.path.exists(lib_path):
                    try:
                        # QUAN TRỌNG: Sử dụng RTLD_GLOBAL để các lib sau có thể nhìn thấy lib này
                        # mode=os.RTLD_GLOBAL hoặc ctypes.RTLD_GLOBAL
                        ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
                        print(f"Loaded dependency: {lib}")
                    except Exception as e:
                        print(f"Failed to load dependency {lib}: {e}")
                        pass
        except Exception as e:
            print(f"Warning: Some dependencies may not be loaded: {e}")
    
    def _init_sdk(self):
        """Khởi tạo SDK"""
        if not self.Objdll.NET_DVR_Init():
            raise Exception("Failed to initialize SDK")
        
        # Set SDK working directory (quan trọng trên Linux)
        if 'linux' in self.platform:
            sdk_path = NET_DVR_LOCAL_SDK_PATH()
            sdk_path.sPath = self.dll_path.encode('utf-8')
            self.Objdll.NET_DVR_SetSDKInitCfg(2, byref(sdk_path))
        
        if not self.Playctrldll.PlayM4_GetPort(byref(self.PlayCtrl_Port)):
            raise Exception("Failed to get playback port")
        
        print("✓ SDK initialized")
    
    def _dec_callback(self, nPort, pBuf, nSize, pFrameInfo, nUser, nReserved2):
        """Decode callback - xử lý frame"""
        if pFrameInfo.contents.nType == 3:  # YUV frame data
            try:
                nWidth = pFrameInfo.contents.nWidth
                nHeight = pFrameInfo.contents.nHeight
                yuv_size = nWidth * nHeight * 3 // 2
                
                yuv_buffer = string_at(pBuf, yuv_size)
                yuv_array = np.frombuffer(yuv_buffer, dtype=np.uint8)
                yuv_frame = yuv_array.reshape((nHeight * 3 // 2, nWidth))
                bgr_frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR_YV12)
                
                with self.frame_lock:
                    self.frame_buffer.append(bgr_frame)
                    self.frame_count += 1
                    
                    self.gc_counter += 1
                    if self.gc_counter >= self.gc_interval:
                        gc.collect()
                        self.gc_counter = 0
                    
            except Exception as e:
                print(f"Error in decode callback: {e}")
    
    def _realdata_callback(self, lPlayHandle, dwDataType, pBuffer, dwBufSize, pUser):
        """Callback nhận stream data"""
        if dwDataType == NET_DVR_SYSHEAD:
            self.Playctrldll.PlayM4_SetStreamOpenMode(self.PlayCtrl_Port, 0)
            
            if self.Playctrldll.PlayM4_OpenStream(
                self.PlayCtrl_Port, pBuffer, dwBufSize, 1024*1024
            ):
                self.FuncDecCB = DECCBFUNWIN(self._dec_callback)
                self.Playctrldll.PlayM4_SetDecCallBackExMend(
                    self.PlayCtrl_Port, self.FuncDecCB, None, 0, None
                )
                
                if self.Playctrldll.PlayM4_Play(self.PlayCtrl_Port, 0):
                    print('✓ Stream playback started')
                else:
                    print('✗ Failed to start playback')
            else:
                print('✗ Failed to open stream')
                
        elif dwDataType == NET_DVR_STREAMDATA:
            self.Playctrldll.PlayM4_InputData(self.PlayCtrl_Port, pBuffer, dwBufSize)
    
    def connect(self, ip, port, username, password):
        """Kết nối đến camera"""
        if self.is_connected:
            print("⚠ Already connected. Disconnect first.")
            return False
        
        print(f"\n[Connect] Connecting to {ip}:{port}...")
        
        struLoginInfo = NET_DVR_USER_LOGIN_INFO()
        struLoginInfo.bUseAsynLogin = 0
        struLoginInfo.sDeviceAddress = bytes(ip, "ascii")
        struLoginInfo.wPort = port
        struLoginInfo.sUserName = bytes(username, "ascii")
        struLoginInfo.sPassword = bytes(password, "ascii")
        struLoginInfo.byLoginMode = 0
        
        struDeviceInfoV40 = NET_DVR_DEVICEINFO_V40()
        self.lUserId = self.Objdll.NET_DVR_Login_V40(
            byref(struLoginInfo), byref(struDeviceInfoV40)
        )
        
        if self.lUserId < 0:
            error_code = self.Objdll.NET_DVR_GetLastError()
            print(f"✗ Connection failed (error code: {error_code})")
            return False
        
        self.is_connected = True
        print(f"✓ Connected successfully (UserID: {self.lUserId})")
        return True
    
    def disconnect(self):
        """Ngắt kết nối"""
        if not self.is_connected:
            print("⚠ Not connected")
            return
        
        if self.is_streaming:
            self.stop_stream()
        
        if self.lUserId >= 0:
            self.Objdll.NET_DVR_Logout(self.lUserId)
            print("✓ Disconnected")
        
        self.lUserId = -1
        self.is_connected = False
    
    def start_stream(self, channel=1, stream_type=0):
        """Bắt đầu streaming"""
        if not self.is_connected:
            print("✗ Not connected. Call connect() first.")
            return False
        
        if self.is_streaming:
            print("⚠ Already streaming. Stop current stream first.")
            return False
        
        print(f"\n[Stream] Starting stream (Channel: {channel}, Type: {stream_type})...")
        
        preview_info = NET_DVR_PREVIEWINFO()
        preview_info.hPlayWnd = 0
        preview_info.lChannel = channel
        preview_info.dwStreamType = stream_type
        preview_info.dwLinkMode = 0
        preview_info.bBlocked = 1
        preview_info.dwDisplayBufNum = 15
        
        self.funcRealDataCallBack = REALDATACALLBACK(self._realdata_callback)
        
        self.lRealPlayHandle = self.Objdll.NET_DVR_RealPlay_V40(
            self.lUserId, byref(preview_info), self.funcRealDataCallBack, None
        )
        
        if self.lRealPlayHandle < 0:
            error_code = self.Objdll.NET_DVR_GetLastError()
            print(f"✗ Failed to start stream (error code: {error_code})")
            return False
        
        self.is_streaming = True
        print("✓ Stream started")
        return True
    
    def stop_stream(self):
        """Dừng streaming"""
        if not self.is_streaming:
            print("⚠ Not streaming")
            return
        
        print("\n[Stop] Stopping stream...")
        
        if self.lRealPlayHandle >= 0:
            self.Objdll.NET_DVR_StopRealPlay(self.lRealPlayHandle)
        
        if self.PlayCtrl_Port.value > -1:
            self.Playctrldll.PlayM4_Stop(self.PlayCtrl_Port)
            self.Playctrldll.PlayM4_CloseStream(self.PlayCtrl_Port)
        
        self.lRealPlayHandle = -1
        self.is_streaming = False
        
        with self.frame_lock:
            self.frame_buffer.clear()
        
        gc.collect()
        print("✓ Stream stopped")
    
    def get_frame(self):
        """Lấy frame mới nhất"""
        with self.frame_lock:
            if len(self.frame_buffer) > 0:
                return self.frame_buffer[-1].copy()
            return None
    
    def is_frame_available(self):
        """Kiểm tra có frame không"""
        with self.frame_lock:
            return len(self.frame_buffer) > 0
    
    def get_frame_count(self):
        """Lấy số frame đã nhận"""
        return self.frame_count
    
    def cleanup(self):
        """Giải phóng tài nguyên"""
        print("\n[Cleanup] Releasing resources...")
        
        if self.is_streaming:
            self.stop_stream()
        
        if self.is_connected:
            self.disconnect()
        
        if self.PlayCtrl_Port.value > -1:
            self.Playctrldll.PlayM4_FreePort(self.PlayCtrl_Port)
            self.PlayCtrl_Port = c_long(-1)
        
        if self.Objdll:
            self.Objdll.NET_DVR_Cleanup()
        
        with self.frame_lock:
            self.frame_buffer.clear()
        
        gc.collect()
        print("✓ Cleanup complete")
    
    def __del__(self):
        """Destructor"""
        self.cleanup()
