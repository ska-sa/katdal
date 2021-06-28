
class TestVisDataV4:
    def test_name_change(self):
        # Ideally this test should do the following
        # >>> d = katdal.open(RDBfile)
        # >>> assert 'URI: ...' in d.__str__()
        # >>> assert 'Name: ...' in d.__str__()
        # but with the limitations in RDBWriter this wasn't feasable
        # See: JIRA SPR1-1152
        pass
