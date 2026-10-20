from panda3d.core import Datagram
from direct.distributed import MsgTypes as A
from direct.distributed.AstronInternalRepository import AstronInternalRepository
from direct.distributed.PyDatagram import PyDatagram
from direct.distributed.PyDatagramIterator import PyDatagramIterator

from direct.distributed.OtpGoDatabaseInterface import OtpGoDatabaseInterface


# This is a mapping of OtpGo message types to their corresponding Astron message types. This is used to translate messages between the two systems.
OUTGOING = {
    A.CONTROL_ADD_CHANNEL: 2001,
    A.CONTROL_REMOVE_CHANNEL: 2002,
    A.CONTROL_ADD_RANGE: 2008,
    A.CONTROL_REMOVE_RANGE: 2009,
    A.CONTROL_ADD_POST_REMOVE: 2010,
    A.CONTROL_CLEAR_POST_REMOVES: 2011,
    A.CONTROL_SET_CON_NAME: 2004,
    A.CONTROL_SET_CON_URL: 2005,
    A.CONTROL_LOG_MESSAGE: 2012,
    A.STATESERVER_CREATE_OBJECT_WITH_REQUIRED: 2001,
    A.STATESERVER_CREATE_OBJECT_WITH_REQUIRED_OTHER: 2003,
    A.STATESERVER_DELETE_AI_OBJECTS: 2061,
    A.STATESERVER_OBJECT_SET_FIELD: 2004,
    A.STATESERVER_OBJECT_SET_FIELDS: 2005,
    A.STATESERVER_OBJECT_DELETE_RAM: 2007,
    A.STATESERVER_OBJECT_SET_LOCATION: 2008,
    A.STATESERVER_OBJECT_SET_AI: 2045,
    A.STATESERVER_OBJECT_SET_OWNER: 2070,
    A.STATESERVER_OBJECT_GET_AI: 2054,
    A.STATESERVER_OBJECT_GET_OWNER: 2064,
    A.STATESERVER_OBJECT_GET_LOCATION: 2022,
    A.STATESERVER_OBJECT_GET_ALL: 2020,
    A.DBSS_OBJECT_GET_ACTIVATED: 2207,
    A.DBSS_OBJECT_DELETE_DISK: 2232,
    A.CLIENTAGENT_SET_STATE: 3000,
    A.CLIENTAGENT_SET_CLIENT_ID: 3001,
    A.CLIENTAGENT_SEND_DATAGRAM: 3002,
    A.CLIENTAGENT_EJECT: 3004,
    A.CLIENTAGENT_DROP: 3005,
    A.CLIENTAGENT_GET_NETWORK_ADDRESS: 3006,
    A.CLIENTAGENT_DECLARE_OBJECT: 3010,
    A.CLIENTAGENT_UNDECLARE_OBJECT: 3011,
    A.CLIENTAGENT_ADD_SESSION_OBJECT: 3012,
    A.CLIENTAGENT_REMOVE_SESSION_OBJECT: 3013,
    A.CLIENTAGENT_SET_FIELDS_SENDABLE: 120,
    A.CLIENTAGENT_OPEN_CHANNEL: 3100,
    A.CLIENTAGENT_CLOSE_CHANNEL: 3105,
    A.CLIENTAGENT_ADD_POST_REMOVE: 3110,
    A.CLIENTAGENT_CLEAR_POST_REMOVES: 3111,
}


def translate(datagram):
    di = PyDatagramIterator(datagram)
    channels = [di.getUint64() for _ in range(di.getUint8())]
    control = channels == [A.CONTROL_CHANNEL]
    sender = None if control else di.getUint64()
    message = di.getUint16()
    native = OUTGOING.get(message)
    if native is None:
        if message >= 20000 or 2100 <= message <= 2124:
            native = message
        else:
            raise NotImplementedError('Unsupported OtpGo message: %d' % message)

    result = PyDatagram()
    result.addUint8(len(channels))
    for channel in channels:
        result.addUint64(4001 if control else channel)
    if not control:
        result.addUint64(sender)
    result.addUint16(native)

    if control and message == A.CONTROL_ADD_POST_REMOVE:
        di.getUint64()  # Astron groups post-removes by sender; OtpGo uses the connection.
        result.addBlob(translate(Datagram(di.getBlob())).getMessage())
    elif control and message == A.CONTROL_CLEAR_POST_REMOVES:
        di.getUint64()
    elif message == A.CLIENTAGENT_ADD_POST_REMOVE:
        result.addBlob(translate(Datagram(di.getBlob())).getMessage())
    elif message in (A.STATESERVER_CREATE_OBJECT_WITH_REQUIRED,
                     A.STATESERVER_CREATE_OBJECT_WITH_REQUIRED_OTHER):
        doId, parent, zone, cls = di.getUint32(), di.getUint32(), di.getUint32(), di.getUint16()
        result.addUint32(parent)
        result.addUint32(zone)
        result.addUint16(cls)
        result.addUint32(doId)
    elif message == A.CLIENTAGENT_SET_FIELDS_SENDABLE:
        result.addUint32(di.getUint32())
        di.getUint16()  # OtpGo uses the packet length rather than a field count.
    result.appendData(di.getRemainingBytes())
    return result


class OtpGoInternalRepository(AstronInternalRepository):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setHandleCUpdates(False)
        self.dbInterface = OtpGoDatabaseInterface(self)

    def sendNative(self, datagram):
        super().send(datagram)

    def send(self, datagram):
        self.sendNative(translate(datagram))

    def sendActivate(self, doId, parentId, zoneId, dclass=None, fields=None):
        if dclass is None:
            def gotObject(cls, values):
                if cls is None:
                    self.notify.error('Cannot activate missing object %d' % doId)
                self.sendActivate(doId, parentId, zoneId, cls, fields)
            self.dbInterface.queryObject(self.dbId, doId, gotObject)
            return
        dg = PyDatagram()
        dg.addServerHeader(doId, self.ourChannel, 2051)
        dg.addUint32(parentId)
        dg.addUint32(zoneId)
        dg.addUint64(0)
        dg.addUint16(dclass.getNumber())
        dg.addUint32(0)
        for i in range(dclass.getNumInheritedFields()):
            field = dclass.getInheritedField(i)
            if field.isRequired() and not field.asMolecularField():
                dg.appendData(field.getDefaultValue())
        fields = fields or {}
        dg.addUint16(len(fields))
        for name, value in fields.items():
            field = dclass.getFieldByName(name)
            dg.addUint16(field.getNumber())
            dg.appendData(self.dbInterface.pack(field, value))
        self.sendNative(dg)

    def handleDatagram(self, di):
        message = self.getMsgType()
        if message in (1004, 1013, 1016):
            self.dbInterface.handleDatagram(message, di)
        elif message == 2004:
            doId = di.getUint32()
            obj = self.doId2do.get(doId)
            if obj:
                obj.dclass.receiveUpdate(obj, di)
        elif message == 2067:
            di.getUint32()
            self.handleObjEntry(self.reorderEntry(di), True)
        elif message in (2007, 2033):
            self.handleObjExit(di)
        elif message == 2009:
            self.handleObjLocation(di)
        elif message == 2208:
            self.handleGetActivatedResp(di)
        elif message == 2023:
            self.handleGetLocationResp(di)
        elif message == 2030:
            context = di.getUint32()
            entry = self.reorderEntry(di)
            dg = PyDatagram()
            dg.addUint32(context)
            dg.appendData(entry.getRemainingBytes())
            self.handleGetObjectResp(PyDatagramIterator(dg))
        elif message == 3007:
            self.handleGetNetworkAddressResp(di)
        elif message >= 20000:
            self.netMessenger.handle(message, di)
        elif message != 2052:  # Activation acknowledgement
            self.notify.warning('Unhandled OtpGo message %d' % message)

    @staticmethod
    def reorderEntry(di):
        parent, zone, cls, doId = di.getUint32(), di.getUint32(), di.getUint16(), di.getUint32()
        dg = PyDatagram()
        dg.addUint32(doId)
        dg.addUint32(parent)
        dg.addUint32(zone)
        dg.addUint16(cls)
        dg.appendData(di.getRemainingBytes())
        return PyDatagramIterator(dg)
