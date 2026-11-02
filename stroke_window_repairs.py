"""Preserve a model's temporal support while correcting its contact center.

No stroke-type label is consulted. Bounds describe proposed model context, not
human-verified preparation/follow-through boundaries. Pure NumPy, no inference.
"""
import numpy as np

VERSION = 'translate-original-contact-window-v1'


def translate_contact_window(dense, event, peak_time):
    times = np.asarray([f['source_time'] for f in dense['frames']],float)
    if not len(times):
        raise ValueError('No decoded frames')
    if not np.isfinite(peak_time) or not times[0] <= peak_time <= float(dense['end']):
        raise ValueError('Contact time outside the decoded recording')
    original_center = int(event.get('frame_index',np.argmin(abs(times-event['peak_time']))))
    center = int(np.argmin(abs(times-peak_time)))
    shift = center-original_center
    a = max(0,int(event['start_frame'])+shift)
    b = min(len(times)-1,int(event['end_frame'])+shift)
    if not a <= center <= b:
        raise ValueError('Original support does not contain its contact center')
    dt = float(np.median(np.diff(times))) if len(times)>1 else 1/float(dense['sample_fps'])
    return {**event,'frame_index':center,'peak_time':float(times[center]),
        'start_frame':a,'end_frame':b,'start':float(times[a]),
        'last_frame_time':float(times[b]),'end':min(float(dense['end']),float(times[b]+dt)),
        'duration':float(min(float(dense['end']),times[b]+dt)-times[a]),
        'basis':'human_contact_translated_original_model_support',
        'window_repair_version':VERSION,'original_center_frame':original_center,
        'translation_frames':shift,'original_support_frame_count':int(event['end_frame'])-int(event['start_frame'])+1,
        'recording_edge_clipped':a != int(event['start_frame'])+shift or b != int(event['end_frame'])+shift,
        'boundary_confirmed':False}
