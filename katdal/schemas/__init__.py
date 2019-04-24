"""Makes packaged XSD schemas available as validators."""

from lxml import etree

import pkg_resources

#def _make_validator(schema):
#    """Check a schema document and create a validator from it"""
#    validator_cls = jsonschema.validators.validator_for(schema)
#    validator_cls.check_schema(schema)
#    return validator_cls(schema, format_checker=jsonschema.FormatChecker())


class ValidatorWithLog(object):
    def __init__(self, validator):
        self.validator = validator

    def validate(self, xml_string):
        """Validates a supplied XML string against
        the instantiated validator.
        Failure to validate will raise DocumentInvalid
        with the output of the error log."""
        try:
            xml_doc = etree.fromstring(bytes(bytearray(xml_string, encoding='utf-8')))
        except etree.XMLSyntaxError as e:
            raise ValueError(e)
        if not self.validator.validate(xml_doc):
            log = self.validator.error_log
            raise etree.DocumentInvalid(log.last_error)
        return True

for name in pkg_resources.resource_listdir(__name__, '.'):
    if name.endswith('.xsd'):
        #with open(pkg_resources.resource_stream(__name__, name)) as f:
        xmlschema_doc = etree.parse(pkg_resources.resource_stream(__name__, name))
        xml_validator = etree.XMLSchema(xmlschema_doc)
        #reader = codecs.getreader('utf-8')(pkg_resources.resource_stream(__name__, name))
        #schema = json.load(reader)
        globals()[name[:-4].upper()] = ValidatorWithLog(xml_validator) #_make_validator(schema)
