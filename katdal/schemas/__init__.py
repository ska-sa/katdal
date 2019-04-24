"""Makes packaged XSD schemas available as validators."""

from lxml import etree

import pkg_resources


class ValidatorWithLog(object):
    def __init__(self, validator):
        self.validator = validator

    def validate(self, xml_string):
        """Validates a supplied XML string against the instantiated validator.

        Parameters
        ---------
        xml_string : str
            String representation of the XML to be turned into a document
            and validated.

        Raises
        ------
        etree.DocumentInvalid
            if `xml_string` does not validate against the XSD schema
        ValueError
            if `xml_string` cannot be parsed into a valid XML document
        """
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
        xmlschema_doc = etree.parse(pkg_resources.resource_stream(__name__, name))
        xml_validator = etree.XMLSchema(xmlschema_doc)
        globals()[name[:-4].upper()] = ValidatorWithLog(xml_validator)
