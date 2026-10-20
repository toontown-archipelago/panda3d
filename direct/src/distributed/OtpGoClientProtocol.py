from direct.distributed import MsgTypes as A
from direct.distributed.PyDatagram import PyDatagram
from direct.distributed.PyDatagramIterator import PyDatagramIterator
from panda3d.direct import DCPacker


MESSAGES = {
    24: A.CLIENT_OBJECT_SET_FIELD,
    25: A.CLIENT_OBJECT_LEAVING,
    26: A.CLIENT_OBJECT_LEAVING_OWNER,
    27: A.CLIENT_OBJECT_LEAVING,
    34: A.CLIENT_ENTER_OBJECT_REQUIRED,
    35: A.CLIENT_ENTER_OBJECT_REQUIRED_OTHER,
    36: A.CLIENT_ENTER_OBJECT_REQUIRED_OTHER_OWNER,
    48: A.CLIENT_DONE_INTEREST_RESP,
    52: A.CLIENT_HEARTBEAT,
    102: A.CLIENT_OBJECT_LOCATION,
}


def decode(repository, message, di):
    if message in (34, 35, 36):
        parent, zone, cls, doId = di.getUint32(), di.getUint32(), di.getUint16(), di.getUint32()
        dg = PyDatagram()
        dg.addUint32(doId)
        dg.addUint32(parent)
        dg.addUint32(zone)
        dg.addUint16(cls)
        if message == 36:
            # We need to unpack the molecular fields and only send the broadcast and ownrecv fields to the client.
            packer = DCPacker()
            packer.setUnpackData(di.getRemainingBytes())
            dclass = repository.dclassesByNumber[cls]
            for i in range(dclass.getNumInheritedFields()):
                field = dclass.getInheritedField(i)
                if field.isRequired() and not field.asMolecularField():
                    start = packer.getNumUnpackedBytes()
                    packer.beginUnpack(field)
                    packer.unpackSkip()
                    packer.endUnpack()
                    if field.isBroadcast() or field.isOwnrecv():
                        dg.appendData(di.getRemainingBytes()[start:packer.getNumUnpackedBytes()])
            dg.appendData(di.getRemainingBytes()[packer.getNumUnpackedBytes():])
        else:
            dg.appendData(di.getRemainingBytes())
        di = PyDatagramIterator(dg)
    elif message == 48:
        handle, context = di.getUint16(), di.getUint32()
        dg = PyDatagram()
        dg.addUint32(context)
        dg.addUint16(handle)
        di = PyDatagramIterator(dg)
    message = MESSAGES.get(message, message)
    dg = PyDatagram()
    dg.addUint16(message)
    dg.appendData(di.getRemainingBytes())
    return message, PyDatagramIterator(dg, 2)
