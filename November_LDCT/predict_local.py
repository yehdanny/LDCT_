#from log import #logger
import os
import sys
import numpy as np
import SimpleITK as sitk
import pydicom
#import dcmtrans
import cv2
import time
import concurrent.futures
#from lungmask import LMInferer

# TensorFlow/Keras imports for ResUNet model
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, BatchNormalization, ReLU, Add, SpatialDropout2D
from scipy import ndimage as ndi
from skimage.filters import roberts
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
from skimage.morphology import convex_hull_image
from skimage.morphology import disk, binary_closing
from collections import defaultdict
# 設定視窗參數
WINDOWS = {'window_center': -400, 'window_width': 1200}

# ResUNet model parameters
PATCH_SIZE = 176
N = 3  # 9 patches
BATCH = 128
THR = 0.96
USE_MMAP = True

# Post-processing parameters
IOU_THR_SLICE = 0.3
CENTER_THR_PX = 12
MAX_Z_GAP = 1
MIN_SLICE_LEN = 2
MIN_AREA_PX = 5 * 5
KEEP_TOPK_PER_SLICE = None

# ResUNet model functions
def load_dicom_series(series_dir):
    """
    用 SimpleITK 讀取單一病例資料夾下的一整個 DICOM series
    回傳：ct (Z, Y, X)、origin(x,y,z)、spacing(x,y,z)
    """
    try:
        # 方法1: 使用 SimpleITK 的標準讀取方式
        series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(series_dir)
        if not series_IDs:
            raise RuntimeError(f"No DICOM series found in: {series_dir}")
        series_files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(series_dir, series_IDs[0])
        
        # 檢查檔案是否存在
        for file_path in series_files:
            if not os.path.exists(file_path):
                #logger.warning(f"DICOM 檔案不存在: {file_path}")
                pass
        
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(series_files)
        image = reader.Execute()
        ct = sitk.GetArrayFromImage(image)
        origin = np.array(list(image.GetOrigin()))
        spacing = np.array(list(image.GetSpacing()))
        return ct, origin, spacing
        
    except Exception as e:
        #logger.error(f"SimpleITK 讀取失敗: {e}")
        
        # 方法2: 使用 pydicom 備用讀取方式
        try:
            #logger.info("嘗試使用 pydicom 備用讀取方式")
            return load_dicom_series_pydicom(series_dir)
        except Exception as e2:
            #logger.error(f"pydicom 備用讀取也失敗: {e2}")
            raise RuntimeError(f"無法讀取 DICOM series: {series_dir}. 錯誤: {e}")

def load_dicom_series_pydicom(series_dir):
    """
    使用 pydicom 備用讀取 DICOM series
    """
    import glob
    
    # 獲取所有 DICOM 檔案
    dicom_files = glob.glob(os.path.join(series_dir, "*.dcm"))
    if not dicom_files:
        dicom_files = glob.glob(os.path.join(series_dir, "*"))
        dicom_files = [f for f in dicom_files if os.path.isfile(f)]
    
    if not dicom_files:
        raise RuntimeError(f"No DICOM files found in: {series_dir}")
    
    # 讀取第一個檔案獲取基本資訊
    first_ds = pydicom.dcmread(dicom_files[0], force=True)
    
    # 讀取所有檔案並排序
    datasets = []
    for file_path in dicom_files:
        try:
            ds = pydicom.dcmread(file_path, force=True)
            datasets.append((ds, file_path))
        except Exception as e:
            #logger.warning(f"無法讀取檔案 {file_path}: {e}")
            continue
    
    if not datasets:
        raise RuntimeError("沒有成功讀取任何 DICOM 檔案")
    
    # 按位置排序
    def get_position(ds):
        try:
            return float(ds.ImagePositionPatient[2])
        except (AttributeError, TypeError, ValueError):
            try:
                return float(ds.SliceLocation)
            except (AttributeError, TypeError, ValueError):
                return 0
    
    datasets.sort(key=lambda x: get_position(x[0]))
    
    # 構建 3D 陣列
    pixel_arrays = []
    for ds, _ in datasets:
        pixel_array = ds.pixel_array
        if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
            pixel_array = pixel_array.astype(np.float32) * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
        pixel_arrays.append(pixel_array)
    
    ct = np.stack(pixel_arrays, axis=0)
    
    # 獲取 origin 和 spacing
    try:
        origin = np.array([float(x) for x in first_ds.ImagePositionPatient])
    except (AttributeError, TypeError, ValueError):
        origin = np.array([0.0, 0.0, 0.0])
    
    try:
        spacing = np.array([float(x) for x in first_ds.PixelSpacing] + [float(first_ds.SliceThickness)])
    except (AttributeError, TypeError, ValueError):
        spacing = np.array([1.0, 1.0, 1.0])
    
    return ct, origin, spacing

def sanitize_hu(vol_np, pixel_padding_value=None):
    v = vol_np.astype(np.float32, copy=True)
    if pixel_padding_value is not None and pixel_padding_value <= -1500:
        v[vol_np <= pixel_padding_value + 1e-6] = -1024.0
    v[v < -1024.0] = -1024.0
    v[v > 3071.0] = 3071.0
    return v

def window_to_uint8(v_hu, wl=-600, ww=1500):
    low, high = wl - ww/2, wl + ww/2
    v = np.clip(v_hu, low, high)
    return ((v - low) / ww * 255).astype('uint8')

def linspace_starts(L, tile, n):
    """在長度 L 上均勻取 n 個起點"""
    if L <= tile:
        return [0]
    return [int(round(i * (L - tile) / (n - 1))) for i in range(n)]

def safe_crop(arr, y0, x0, size):
    """從 (y0,x0) 切 size×size；若越界則零填充"""
    H, W = arr.shape
    y1, x1 = y0 + size, x0 + size
    y0c, x0c = max(0, y0), max(0, x0)
    y1c, x1c = min(H, y1), min(W, x1)
    crop = arr[y0c:y1c, x0c:x1c]
    pad_top = y0c - y0
    pad_left = x0c - x0
    pad_bot = y1 - y1c
    pad_right = x1 - x1c
    if pad_top or pad_left or pad_bot or pad_right:
        crop = np.pad(crop, ((pad_top, pad_bot), (pad_left, pad_right)),
                      mode='constant', constant_values=0)
    return crop

def res_block(x, filters, wd=1e-4, drop=0.0, dilation_rate=(1, 1)):
    """Residual block with optional dilation rate"""
    shortcut = x
    in_ch = K.int_shape(x)[-1]

    x = Conv2D(filters, 3, padding="same", dilation_rate=dilation_rate, use_bias=False,
               kernel_initializer="he_normal", kernel_regularizer=l2(wd))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(filters, 3, padding="same", dilation_rate=dilation_rate, use_bias=False,
               kernel_initializer="he_normal", kernel_regularizer=l2(wd))(x)
    x = BatchNormalization()(x)

    if in_ch != filters:
        shortcut = Conv2D(filters, 1, padding="same", use_bias=False,
                          kernel_initializer="he_normal", kernel_regularizer=l2(wd))(shortcut)
        shortcut = BatchNormalization()(shortcut)
    x = Add()([x, shortcut])
    x = ReLU()(x)

    if drop > 0:
        x = SpatialDropout2D(drop)(x)
    return x

def resunet(input_size=(176, 176, 1), wd=1e-4, drop=0.2):
    """ResUNet model architecture"""
    inputs = Input(input_size)

    c1 = res_block(inputs, 32, wd, drop=0.0)
    p1 = MaxPooling2D(2)(c1)

    c2 = res_block(p1, 64, wd, drop=0.0)
    p2 = MaxPooling2D(2)(c2)

    c3 = res_block(p2, 128, wd, drop=0.0)
    p3 = MaxPooling2D(2)(c3)

    c4 = res_block(p3, 256, wd, drop=drop)
    p4 = MaxPooling2D(2)(c4)

    c5 = res_block(p4, 512, wd, drop=drop, dilation_rate=(2, 2))

    u6 = concatenate([Conv2DTranspose(256, 2, strides=2, padding="same")(c5), c4])
    c6 = res_block(u6, 256, wd, drop=drop)

    u7 = concatenate([Conv2DTranspose(128, 2, strides=2, padding="same")(c6), c3])
    c7 = res_block(u7, 128, wd, drop=drop)

    u8 = concatenate([Conv2DTranspose(64, 2, strides=2, padding="same")(c7), c2])
    c8 = res_block(u8, 64, wd, drop=drop)

    u9 = concatenate([Conv2DTranspose(32, 2, strides=2, padding="same")(c8), c1])
    c9 = res_block(u9, 32, wd, drop=0.0)

    outputs = Conv2D(1, 1, activation="sigmoid", kernel_initializer="he_normal")(c9)
    model = Model(inputs, outputs, name="ResUNet")

    opt = Adam(learning_rate=1e-4)
    model.compile(optimizer=opt, loss=focal_tversky_loss, metrics=[focal_tversky_loss])
    return model

def tversky(y_true, y_pred, alpha=0.7, beta=0.3, smooth=1.0):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    tp = K.sum(y_true * y_pred)
    fp = K.sum((1-y_true) * y_pred)
    fn = K.sum(y_true * (1-y_pred))
    return (tp + smooth) / (tp + alpha*fp + beta*fn + smooth)

def focal_tversky_loss(y_true, y_pred, gamma=0.75):
    return K.pow(1.0 - tversky(y_true, y_pred), gamma)

def get_base_path():
    """獲取基礎路徑，處理打包和未打包的情況"""
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))

def get_model_path():
    """獲取 ResUNet 模型路徑，處理多種可能的路徑情況"""
    base_path = get_base_path()
    possible_paths = [
        os.path.join(base_path, 'model', 'Resunet_model.h5'),
        os.path.join(base_path, '_internal', 'core', 'model', 'Resunet_model.h5'),
        os.path.join(base_path, 'core', 'model', 'Resunet_model.h5'),
        os.path.join(os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else base_path, 
                    'core', 'model', 'Resunet_model.h5')
    ]
    
    for path in possible_paths:
        #logger.info(f'檢查 ResUNet 模型路徑: {path}')
        if os.path.exists(path):
            #logger.info(f'找到 ResUNet 模型: {path}')
            return path
            
    #logger.error('找不到 ResUNet 模型檔案')
    return None

# 確保 sys.stderr 存在
if sys.stderr is None:
    sys.stderr = open(os.devnull, 'w')

def ensure_slice_assets(z_idx: int, ct_norm, case_slice_img, case_lungs):
    """確保某個 z 的 (img_norm, extracted_lungs) 已計算並快取"""
    if z_idx in case_slice_img:
        return

    ct_org = ct_norm[z_idx]
    im = ct_org.copy()
    
    # Step 1: Convert into a binary image
    binary_thr = im < 175
    # Step 2: Remove the blobs connected to the border
    cleared = clear_border(binary_thr)
    # Step 3: Label the image
    label_image = label(cleared)
    # Step 4: Keep the labels with 2 largest areas
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    labels = []
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                    label_image[coordinates[0], coordinates[1]] = 0
            else:
                coordinates = region.coords[0]
                labels.append(label_image[coordinates[0], coordinates[1]])
    else:
        labels = [1, 2]
    
    # Step 5: Fill holes
    right_lung = label_image == labels[0]
    left_lung = label_image == labels[1]
    r_edges = roberts(right_lung)
    l_edges = roberts(left_lung)
    right_lung = ndi.binary_fill_holes(r_edges)
    left_lung = ndi.binary_fill_holes(l_edges)
    # Step 6: convex hull
    right_lung = convex_hull_image(right_lung)
    left_lung = convex_hull_image(left_lung)
    # Step 7: joint two lungs
    sum_of_lr = right_lung + left_lung
    binary = sum_of_lr > 0
    # Step 8: Closure operation
    selem = disk(10)
    binary_c = binary_closing(binary, selem)
    # Step 9: Apply mask
    get_high_vals = binary_c == 0
    im[get_high_vals] = 0
    
    case_slice_img[z_idx] = ct_org.copy()
    case_lungs[z_idx] = im.copy()

class DCM_DATA:
    """處理單個 DICOM 檔案的類別"""
    def __init__(self, filepath):
        self.filepath = filepath
        self.dcmobj = pydicom.dcmread(filepath, force=True)
        if not hasattr(self.dcmobj.file_meta, 'TransferSyntaxUID'):
            self.dcmobj.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        pixel = dcmtrans.read_pixel(filepath, return_on_fail=None)
        imgs, _, _ = dcmtrans.dcmtrans(self.dcmobj, pixel, window=WINDOWS, depth=256)
        self.img = imgs[0]
        self.pixel_spacing = self.dcmobj.PixelSpacing
        self.window_center, self.window_width = self.dcmobj.WindowCenter, self.dcmobj.WindowWidth
        self.slice_thickness = self.dcmobj.SliceThickness
        # print(f"Loaded img shape from {filepath}: {type(self.img)}, shape={getattr(self.img, 'shape', 'N/A')}")

class DCM_DATA_LIST:
    """處理多個 DICOM 檔案的類別"""
    def __init__(self, filepath_list):
        self.filepath_list = filepath_list
        # 使用多線程加速 DICOM 讀取
        with concurrent.futures.ThreadPoolExecutor() as executor:
            self.dcm_data_objs = list(executor.map(DCM_DATA, filepath_list))
            
        # 依照 z 軸位置排序
        def get_z(dcm_data):
            try:
                return float(dcm_data.dcmobj.ImagePositionPatient[2])
            except Exception:
                try:
                    return float(dcm_data.dcmobj.SliceLocation)
                except Exception:
                    return 0
                    
        self.dcm_data_objs.sort(key=get_z)
        self.dcmobj_list = [d.dcmobj for d in self.dcm_data_objs]
        self.img_list = [d.img for d in self.dcm_data_objs]
        self.imgs = [d.img for d in self.dcm_data_objs]
        self.filepath_list = [d.filepath for d in self.dcm_data_objs]
        self.pixel_spacing = self.dcmobj_list[0].PixelSpacing
        self.window_center, self.window_width = self.dcmobj_list[0].WindowCenter, self.dcmobj_list[0].WindowWidth
        
        # 計算切片厚度
        self.slice_thickness = float(self.dcmobj_list[0].SliceThickness)
        
        # 準備 HU 影像
        self.hu_images = []
        for ds in self.dcmobj_list:
            raw = ds.pixel_array
            slope = float(ds.RescaleSlope) if 'RescaleSlope' in ds else 1.0
            intercept = float(ds.RescaleIntercept) if 'RescaleIntercept' in ds else 0.0
            hu_img = raw.astype(np.float32) * slope + intercept
            self.hu_images.append(hu_img)
            
        # 執行肺部分割
        self._perform_lung_segmentation()
        
        # 計算堆疊大小
        self.stack_size = 20  # 預設堆疊大小
        self.half_stack = int(self.stack_size / 2 / self.slice_thickness + 0.5)

    def _perform_lung_segmentation(self):
        """執行 3D 肺部分割"""
        try:
            # 準備 3D 影像
            hu_volume = np.stack(self.hu_images, axis=0)
            sitk_image = sitk.GetImageFromArray(hu_volume)
            
            # 執行分割
            inferer = LMInferer(modelname="R231", tqdm_disable=True)
            self.segmentation_3d = inferer.apply(sitk_image)
            
            # 應用遮罩
            self.masked_images = [np.where(mask == 0, 0, img) 
                                for img, mask in zip(self.img_list, self.segmentation_3d)]
        except Exception as e:
            #logger.error(f"肺部分割失敗: {e}")
            self.masked_images = self.img_list
            self.segmentation_3d = [np.ones_like(img) for img in self.img_list]

    def get_max_projection(self, center_idx):
        """獲取指定切片周圍的最大投影"""
        start = max(0, center_idx - self.half_stack)
        end = min(len(self.img_list), center_idx + self.half_stack + 1)
        
        # 堆疊影像
        stack = np.stack(self.img_list[start:end], axis=0)
        max_proj = np.max(stack, axis=0)
        
        # 堆疊遮罩
        stack_mask = np.stack(self.segmentation_3d[start:end], axis=0)
        max_mask = np.max(stack_mask, axis=0)
        
        # 應用遮罩
        masked_proj = max_proj.copy()
        masked_proj[max_mask == 0] = 0
        
        # # 確保影像格式正確
        # masked_proj = masked_proj.astype(np.uint8)
        # imgs,_,_ = dcmtrans.dcmtrans(self.dcmobj_list[center_idx], max_proj, window=WINDOWS, depth=256)
        # max_proj = imgs[0]
        # imgs,_,_ = dcmtrans.dcmtrans(self.dcmobj_list[center_idx], masked_proj, window=WINDOWS, depth=256)
        # masked_proj = imgs[0]
        
        # 調試用：顯示影像
        # plt.figure(figsize=(15, 5))
        
        # plt.subplot(131)
        # plt.imshow(max_proj, cmap='gray')
        # plt.title(f'Max Projection (Slice {center_idx})')
        # plt.axis('off')
        
        # plt.subplot(132)
        # plt.imshow(masked_proj, cmap='gray')
        # plt.title('Masked Projection')
        # plt.axis('off')
        
        # plt.subplot(133)
        # plt.imshow(max_mask, cmap='gray')
        # plt.title('Lung Mask')
        # plt.axis('off')
        
        # plt.tight_layout()
        # plt.savefig(f'debug/debug_slice_{center_idx:03d}.png')
        # plt.close()
        
        return max_proj, masked_proj

    def get_pixel_spacing(self):
        return self.pixel_spacing
        
    def get_window(self):
        return self.window_center, self.window_width
        
    def len(self):
        return len(self.filepath_list)
        
    def get_sitk_image(self, index):
        return self.imgs[index]
        
    def get_dcmobj(self, index):
        return self.dcmobj_list[index]
        
    def get_img(self, index):
        return self.img_list[index]

def convert_to_serializable(obj):
    """將物件轉換為可序列化的格式"""
    if hasattr(obj, 'to_dict'):
        return obj.to_dict('records')
    elif isinstance(obj, (list, tuple)):
        return list(obj)
    elif isinstance(obj, (int, float, str)):
        return obj
    else:
        return str(obj)

class LDCT_predict:
    """LDCT 預測主類別 - 使用 ResUNet 模型"""
    def __init__(self):
        self.model_path = get_model_path()
        if self.model_path is None:
            raise RuntimeError("無法找到 ResUNet 模型檔案")
        
        #logger.info(f'使用 ResUNet 模型: {self.model_path}')
        print(f'使用 ResUNet 模型: {self.model_path}')
        
        # 載入 ResUNet 模型
        self.model = resunet(input_size=(176, 176, 1))
        self.model.load_weights(self.model_path)
        self.model.compile(optimizer=Adam(learning_rate=2e-4),
                          loss=focal_tversky_loss,
                          metrics=[tversky, focal_tversky_loss, 'binary_accuracy'])
        
        # 初始化批次處理變數
        self.batch_imgs = []
        self.batch_meta = []
        self.slice_nodules = defaultdict(list)
    
    def flush_batch(self):
        """處理批次預測"""
        if not self.batch_imgs:
            return
        
        X = np.stack(self.batch_imgs, axis=0)[..., None].astype(np.float32)
        probs = self.model(X, training=False).numpy()[..., 0]

        for prob, (case_key, z, y0, x0, H, W) in zip(probs, self.batch_meta):
            m = (prob >= THR).astype(np.uint8)
            if m.max() == 0:
                continue
            num, labels, stats, cents = cv2.connectedComponentsWithStats(m, connectivity=8)
            for k in range(1, num):
                x1 = int(stats[k, cv2.CC_STAT_LEFT])
                y1 = int(stats[k, cv2.CC_STAT_TOP])
                w = int(stats[k, cv2.CC_STAT_WIDTH])
                h = int(stats[k, cv2.CC_STAT_HEIGHT])
                # map tile -> slice
                xg1 = int(x0 + x1)
                yg1 = int(y0 + y1)
                xg2 = int(min(xg1 + w, W))
                yg2 = int(min(yg1 + h, H))
                conf = float(prob[labels == k].mean())
                self.slice_nodules[(case_key, z)].append([int(z), xg1, yg1, int(xg2 - xg1), int(yg2 - yg1), conf])
        
        self.batch_imgs.clear()
        self.batch_meta.clear()
    
    def to_xyxy(self, b):
        """轉換邊界框格式"""
        z, x1, y1, w, h, conf = b
        return int(z), int(x1), int(y1), int(x1+w), int(y1+h), float(conf)

    def iou_xyxy(self, a, b):
        """計算兩個邊界框的 IoU"""
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
        inter = iw * ih
        area_a = max(0, ax2-ax1) * max(0, ay2-ay1)
        area_b = max(0, bx2-bx1) * max(0, by2-by1)
        union = area_a + area_b - inter + 1e-6
        return inter / union

    def center_xyxy(self, a):
        """計算邊界框中心點"""
        x1, y1, x2, y2 = a
        return (0.5*(x1+x2), 0.5*(y1+y2))

    def per_slice_nms(self, slice_nodules):
        """每個切片進行 NMS"""
        out = defaultdict(list)
        for (case_key, z), boxes in slice_nodules.items():
            items = []
            for b in boxes:
                z_i, x1, y1, x2, y2, conf = self.to_xyxy(b)
                area = (x2-x1)*(y2-y1)
                if area < MIN_AREA_PX:
                    continue
                items.append([x1, y1, x2, y2, conf])
            if not items:
                            continue
            items.sort(key=lambda t: t[4], reverse=True)
            keep = []
            while items:
                x1, y1, x2, y2, conf = items.pop(0)
                keep.append([x1, y1, x2, y2, conf])
                items = [it for it in items if self.iou_xyxy((x1, y1, x2, y2), it[:4]) < IOU_THR_SLICE]
            if KEEP_TOPK_PER_SLICE is not None:
                keep = keep[:KEEP_TOPK_PER_SLICE]
            for x1, y1, x2, y2, conf in keep:
                out[(case_key, z)].append([int(z), int(x1), int(y1), int(x2-x1), int(y2-y1), float(conf)])
        return out

    def link_across_slices(self, slice_nodules_dict):
        """跨切片連結檢測結果"""
        clusters = []
        by_case = defaultdict(list)
        for (case_key, z), boxes in slice_nodules_dict.items():
            for b in boxes:
                z_i, x1, y1, w, h, conf = b
                by_case[case_key].append((int(z_i), int(x1), int(y1), int(w), int(h), float(conf)))
        
        for case_key, items in by_case.items():
            items.sort(key=lambda t: (t[0], -t[5]))
            tracks = []
            for z, x1, y1, w, h, conf in items:
                x2, y2 = x1+w, y1+h
                cx, cy = 0.5*(x1+x2), 0.5*(y1+y2)
                placed = False
                best_i = -1
                best_cost = 1e9
                for i, tr in enumerate(tracks):
                    if abs(z - tr['last_z']) > MAX_Z_GAP:
                        continue
                    tcx, tcy = self.center_xyxy(tr['last_box'])
                    dz = abs(z - tr['last_z'])
                    dxy = np.hypot(cx - tcx, cy - tcy)
                    if dxy <= CENTER_THR_PX:
                        cost = dxy + 5*dz
                        if cost < best_cost:
                            best_cost = cost
                            best_i = i
                if best_i >= 0:
                    tr = tracks[best_i]
                    tr['last_z'] = z
                    tr['last_box'] = (x1, y1, x2, y2)
                    tr['zs'].append(z)
                    tr['boxes'].append((z, x1, y1, w, h, conf))
                    tr['max_conf'] = max(tr['max_conf'], conf)
                    placed = True
                if not placed:
                    tracks.append(dict(
                        last_z=z, last_box=(x1, y1, x2, y2),
                        zs=[z], boxes=[(z, x1, y1, w, h, conf)], max_conf=conf
                    ))
            for tr in tracks:
                if len(tr['zs']) < MIN_SLICE_LEN:
                    continue
                rep = max(tr['boxes'], key=lambda b: b[-1])
                clusters.append(dict(case=case_key, zs=tr['zs'], boxes=tr['boxes'],
                                   max_conf=tr['max_conf'], rep=rep))
        return clusters
    


    def predict(self, filename_list):
        """執行 LDCT 預測流程 - 使用 ResUNet 模型"""
        try:
            start_time = time.time()
            
            # 獲取第一個檔案的目錄作為病例目錄
            case_dir = os.path.dirname(filename_list[0])
            case_id = os.path.basename(case_dir)
            
            # 載入 DICOM series
            ct, origin, spacing = load_dicom_series(case_dir)
            num_z, H, W = ct.shape
            ct = ct[::-1].copy()  # flip Z axis
            ct = sanitize_hu(ct)
            ct_norm = window_to_uint8(ct, wl=-600, ww=1500)
            
            # 初始化快取
            case_slice_img = {}
            case_lungs = {}
            
            # 處理每個切片
            for idx in range(0, num_z):
                ensure_slice_assets(idx, ct_norm, case_slice_img, case_lungs)
            
            # 準備 tiles 和 infos
            tiles = []
            infos = []
            
            for z in sorted(case_lungs.keys()):
                lungs_out = case_lungs[z].astype(np.uint8)
                H, W = lungs_out.shape
                ys = linspace_starts(H, PATCH_SIZE, N)
                xs = linspace_starts(W, PATCH_SIZE, N)

                for ri, y0 in enumerate(ys):
                    for ci, x0 in enumerate(xs):
                        tile = safe_crop(lungs_out, y0, x0, PATCH_SIZE)
                        tiles.append(tile)
                        
                        y1 = min(y0 + PATCH_SIZE, H)
                        x1 = min(x0 + PATCH_SIZE, W)
                        info = {
                            "case_dir": str(case_id),
                            "z_index": int(z),
                            "r": int(ri),
                            "c": int(ci),
                            "tile_size": int(PATCH_SIZE),
                            "slice_size": {"H": int(H), "W": int(W)},
                            "slice_coords": {"y0": int(y0), "x0": int(x0), "y1": int(y1), "x1": int(x1)},
                            "padded": bool((y1 - y0 < PATCH_SIZE) or (x1 - x0 < PATCH_SIZE)),
                        }
                        infos.append(info)
            
            # 批次處理預測
            self.slice_nodules = defaultdict(list)
            for info, tile in zip(infos, tiles):
                z = int(info["z_index"])
                y0 = int(info["slice_coords"]["y0"])
                x0 = int(info["slice_coords"]["x0"])
                H = int(info["slice_size"]["H"])
                W = int(info["slice_size"]["W"])
                case_key = info["case_dir"]

                if USE_MMAP:
                    if tile.max() == 0:
                        continue
                    tile = tile.astype(np.float32) / 255.0
                else:
                    tile = tile.astype(np.float32) / 255.0
                    if tile.max() == 0:
                        continue

                self.batch_imgs.append(tile)
                self.batch_meta.append((case_key, z, y0, x0, H, W))
                
                if len(self.batch_imgs) >= BATCH:
                    self.flush_batch()
            
            # 處理剩餘的批次
            self.flush_batch()
            
            # 後處理
            slice_nodules_nms = self.per_slice_nms(self.slice_nodules)
            clusters = self.link_across_slices(slice_nodules_nms)
            
            # 轉換為原始格式
            predict_class = []
            predict_prob = []
            predict_coord = []
            
            for cluster in clusters:
                nodules = cluster['boxes']
                for rep in nodules :
                    z, x1, y1, w, h, conf = rep
                    predict_class.append(1)  # 檢測到結節
                    predict_prob.append(conf)
                    predict_coord.append({
                        'filepath': filename_list[min(z, len(filename_list)-1)],
                        'xmin': x1,
                        'ymin': y1,
                        'xmax': x1 + w,
                        'ymax': y1 + h,
                        'confidence': conf,
                        'class': 0,
                        'name': 'nodule'
                    })
            
            end_time = time.time()
            #logger.info(f"ResUNet 預測時間: {end_time - start_time} 秒")
            print(f"ResUNet 預測時間: {end_time - start_time} 秒")
                
            # 整理結果
            result = {
                "predict_class": convert_to_serializable(predict_class),
                "predict_prob": convert_to_serializable(predict_prob),
                "predict_coord": convert_to_serializable(predict_coord)
            }
            
            return result
            
        except Exception as e:
            #logger.error(f"預測過程中發生錯誤: {e}")
            return {
                "predict_class": [],
                "predict_prob": [],
                "predict_coord": []
            }
if __name__ == "__main__":
    # 測試 LDCT_predict 類別
    test_case_dir = r"D:\Daniel\for_git\LDCT_git\5407878_20240125"  # 替換為實際的 DICOM series 路徑
    dicom_files = [os.path.join(test_case_dir, f) for f in os.listdir(test_case_dir) if f.endswith('.dcm')]
    
    predictor = LDCT_predict()
    results = predictor.predict(dicom_files)
    
    print("預測結果:")
    print(results)