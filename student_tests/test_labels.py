import json
import pytest
from student_vlm.labels import (ground_truth, iou_reward, objects_to_answer,
                                prompt_for, yolo_to_box)


def test_grounding_conversion_and_clipping():
    assert yolo_to_box('crack', [.5,.5,.2,.4])['bbox_2d'] == [400,300,600,700]
    assert yolo_to_box('pothole', [0,1,.2,.2])['bbox_2d'] == [0,900,100,1000]
    assert yolo_to_box('crack', [.5,.5,.2,.4], coord='pixel')['bbox_2d'] == [256,108,384,252]


def test_stable_top_left_answer_order_and_empty():
    objects = [{'label':'crack','yolo':[.8,.8,.1,.1]},
               {'label':'pothole','yolo':[.2,.2,.1,.1]}]
    assert json.loads(objects_to_answer(objects))[0]['label'] == 'pothole'
    assert objects_to_answer([]) == '[]'
    assert '0 to 1000' in prompt_for() and 'pixel' in prompt_for('pixel')


@pytest.mark.parametrize('label,yolo', [('tree',[.5]*4),('crack',[float('nan')]*4),
                                      ('crack',[-.1,.5,.1,.1])])
def test_bad_labels_fail_before_training(label,yolo):
    with pytest.raises(ValueError): yolo_to_box(label,yolo)


GT=[{'label':'pothole','bbox':(100,100,300,300)}]
GOOD='[{"label":"pothole","bbox_2d":[100,100,300,300]}]'


def test_reward_perfect_invalid_wrong_class_and_empty():
    assert iou_reward(GOOD,GT) == 1
    assert iou_reward('not json',GT) == 0
    assert iou_reward(GOOD.replace('pothole','crack'),GT) == .2
    assert iou_reward('[]',[]) == 1
    assert iou_reward('[]',GT) == .2
    assert iou_reward(GOOD,[]) == .2


def test_reward_duplicate_and_giant_box_cannot_win():
    assert iou_reward(json.dumps(json.loads(GOOD)*2),GT) == pytest.approx(.6)
    giant='[{"label":"pothole","bbox_2d":[0,0,1000,1000]}]'
    assert iou_reward(giant,GT) == .2
    assert ground_truth([{'label':'pothole','yolo':[.2,.2,.2,.2]}])[0]['bbox'] == pytest.approx(GT[0]['bbox'])
