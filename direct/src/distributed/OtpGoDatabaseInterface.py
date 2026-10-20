from panda3d.direct import DCPacker
from direct.distributed.PyDatagram import PyDatagram


class OtpGoDatabaseInterface:
    def __init__(self, air):
        self.air = air
        self.callbacks = {}

    @staticmethod
    def pack(field, value):
        packer = DCPacker()
        packer.beginPack(field)
        field.packArgs(packer, value)
        if not packer.endPack():
            raise ValueError('Invalid value for %s' % field.getName())
        return packer.getBytes()

    @staticmethod
    def unpack(field, data):
        packer = DCPacker()
        packer.setUnpackData(data)
        packer.beginUnpack(field)
        value = field.unpackArgs(packer)
        if not packer.endUnpack():
            raise ValueError('Invalid data for %s' % field.getName())
        return value

    def request(self, databaseId, message, callback):
        context = self.air.getContext()
        self.callbacks[context] = callback
        dg = PyDatagram()
        dg.addServerHeader(databaseId, self.air.ourChannel, message)
        dg.addUint32(context)
        return dg

    def createObject(self, databaseId, dclass, fields=None, callback=None):
        dg = self.request(databaseId, 1003, lambda di: self.created(di, callback))
        dg.addString('')
        dg.addUint16(dclass.getNumber())
        fields = fields or {}
        dg.addUint16(len(fields))
        for name, value in fields.items():
            dg.addString(name)
            dg.addBlob(self.pack(dclass.getFieldByName(name), value))
        self.air.sendNative(dg)

    @staticmethod
    def created(di, callback):
        status = di.getUint8()
        doId = di.getUint32()
        if callback:
            callback(0 if status else doId)

    def getValues(self, databaseId, doId, names, callback):
        dg = self.request(databaseId, 1012, lambda di: self.received(di, callback))
        dg.addUint32(doId)
        dg.addUint16(len(names))
        for name in names:
            dg.addString(name)
        self.air.sendNative(dg)

    @staticmethod
    def received(di, callback):
        di.getUint32()  # Object ID.
        names = [di.getString() for _ in range(di.getUint16())]
        if di.getUint8():
            callback(None)
            return
        values = {}
        for name in names:
            data = di.getBlob()
            if di.getUint8():
                values[name] = data
        callback(values)

    def queryObject(self, databaseId, doId, callback, dclass=None, fieldNames=()):
        if dclass is None:
            def gotClass(values):
                if not values or 'DcObjectType' not in values:
                    callback(None, None)
                    return
                from direct.distributed.PyDatagramIterator import PyDatagramIterator
                from panda3d.core import Datagram
                name = PyDatagramIterator(Datagram(values['DcObjectType'])).getString()
                cls = self.air.dclassesByName.get(name + self.air.dcSuffix)
                if cls is None:
                    callback(None, None)
                    return
                self.queryObject(databaseId, doId, callback, cls, fieldNames)
            self.getValues(databaseId, doId, ['DcObjectType'], gotClass)
            return

        names = list(fieldNames) or [
            dclass.getInheritedField(i).getName()
            for i in range(dclass.getNumInheritedFields())
            if dclass.getInheritedField(i).isDb()
        ]

        def gotFields(values):
            if values is None:
                callback(None, None)
                return
            fields = {}
            for name in names:
                field = dclass.getFieldByName(name)
                data = values.get(name, field.getDefaultValue())
                fields[name] = self.unpack(field, data)
            callback(dclass, fields)

        self.getValues(databaseId, doId, names, gotFields)

    def updateObject(self, databaseId, doId, dclass, newFields, oldFields=None, callback=None):
        if oldFields is not None:
            if set(oldFields) != set(newFields):
                raise ValueError('Conditional updates must contain the same fields')
            def updated(di):
                status = di.getUint8()
                result = None
                if status == 1:
                    result = {}
                    for _ in range(di.getUint16()):
                        name = di.getString()
                        result[name] = self.unpack(dclass.getFieldByName(name), di.getBlob())
                elif status:
                    result = {'error': 'OtpGo conditional database update failed'}
                if callback:
                    callback(result)
            dg = self.request(databaseId, 1015, updated)
        else:
            dg = PyDatagram()
            dg.addServerHeader(databaseId, self.air.ourChannel, 1014)
        dg.addUint32(doId)
        dg.addUint16(len(newFields))
        for name, value in newFields.items():
            dg.addString(name)
            if oldFields is not None:
                dg.addBlob(self.pack(dclass.getFieldByName(name), oldFields[name]))
            dg.addBlob(self.pack(dclass.getFieldByName(name), value))
        self.air.sendNative(dg)
        if callback and oldFields is None:
            callback(None)

    def handleDatagram(self, message, di):
        context = di.getUint32()
        callback = self.callbacks.pop(context, None)
        if callback:
            callback(di)
