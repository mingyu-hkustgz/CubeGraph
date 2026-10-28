"""Independent classification; invalid results are never accepted as valid cells."""
import numpy as np


def check_ids(actual, gt, object_count, qualifies):
    assert actual.shape == gt.shape
    hits = violations = 0
    examples = []
    for qi, row in enumerate(actual):
        assert np.all(row >= -1), (qi, 'invalid negative ID')
        got = row[row >= 0]
        assert np.all(got < object_count), (qi, 'out-of-range ID')
        assert len(set(got)) == len(got), (qi, 'duplicate ID')
        eligible = qualifies(qi, got)
        bad = got[~eligible]
        violations += len(bad)
        for object_id in bad:
            if len(examples) < 20:
                examples.append(dict(query=qi, object_id=int(object_id)))
        hits += len(set(got) & set(gt[qi][gt[qi] >= 0]))
    denominator = int((gt >= 0).sum())
    return dict(valid=violations == 0, predicate_violations=violations,
                recall=hits / denominator if denominator else 1., examples=examples,
                result_status='PASS' if violations == 0 else 'FAIL_PREDICATE')


def selftest():
    actual = np.array([[0, 1, -1], [2, 3, -1]])
    gt = actual.copy()
    valid = check_ids(actual, gt, 4, lambda qi, ids: np.ones(len(ids), dtype=bool))
    assert valid['valid'] and valid['recall'] == 1
    invalid = check_ids(actual, gt, 4, lambda qi, ids: ids != 2)
    assert not invalid['valid'] and invalid['predicate_violations'] == 1
    assert invalid['result_status'] == 'FAIL_PREDICATE'
    for bad in (np.array([[0, 0, -1], [2, 3, -1]]),
                np.array([[0, 4, -1], [2, 3, -1]]),
                np.array([[0, -2, -1], [2, 3, -1]])):
        try:
            check_ids(bad, gt, 4, lambda qi, ids: np.ones(len(ids), dtype=bool))
        except AssertionError:
            continue
        raise AssertionError('invalid structural output accepted')
    print('VALIDITY_SELFTEST_PASS')


if __name__ == '__main__':
    selftest()
