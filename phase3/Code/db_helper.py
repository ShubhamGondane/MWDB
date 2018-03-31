from qpython import qconnection

[staticmethod]


# Connect to database
def db_connect(host="localhost", port=5000):
    try:
        q = qconnection.QConnection(host="localhost", port=5000)
        q.open()
        return q
    except Exception as err:
        print "Database connection failed."
        print err
