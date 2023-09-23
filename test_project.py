from project import load_data, get_net_from_file, net_report

def test_load_data():
    data, _ = load_data()
    assert data[0][1][5] == 1
    
def test_get_net_from_file():
    net = get_net_from_file()
    assert net.shape == (784,30,10)
    
def test_net_report():
    report = net_report((2,4,2), 1, 100, 10, 10, 20)
    report_check = """
TRAINING DETAILS:
    Network Shape:          (2, 4, 2)
    Learning Coefficient:   1.00
    Data Size:              100
    Batch Size:             10
    Repetitions:            10

RESULTS:
    Rate:                   20.00%
    Score:                  20/100
"""
    assert report == report_check