import pytest

from sample import main


@pytest.mark.timeout(3)
def test_sample_main():
    data_x = [1.1, 2, 3, 4, 5, 6, 7, 8.5, 9, 10, 11, 12, 13.1, 14, 15]
    data_y = [2, 5.2, 7.1, 9.2, 11.1, 13.2, 15, 17.1, 18, 21.2, 23.1, 25, 27.1, 29, 30.2]

    _, slope, bias = main(data_x, data_y)
    assert 1.8 <= slope and slope <= 2.2
    assert -1 <= bias and bias <= 1
