import discord
import talk 
from random import choice
client = discord.Client()
bot = talk.talkWithMe()
#Token
token = "NDM4OTI5OTkzMjIyMDYyMTAw.DcLxGg.QsH7hx4fyLkBU1ToCpJdtTomaxU"

#Type of Greeting
greeting = ['hello','hi','สวัสดี','ดี','หวัดดี','ไง']

#Type of Jokes
joke = ['ขอมุก', 'ขอมุข', 'มุกตลก']
jokes = ['ไม่อยากอยู่ใกล้ ไม่อยากอยู่ไกล อยากอยู่ในใจของเธอได้เปล่า', 'คนดี คนชั่ว มีอยู่ทั่วไปแต่คนที่ดีกับใจ💖 มีแค่เทอคนเดียวว', 'ร้อนก็เปิดแอร์ อยากได้คนเทคแคร์ก็ทักมาา 555', 'เบื่อแล้ว เพลงที่มีงู ขอเป็นเพลงที่มี you แทนได้ปะ 😚😂😚', 'อยากให้เธอเมา เมาที่รอง มองที่เรา 😜😜😜', 'รักชาติให้ยืนตรง..แต่ถ้าอยากรักมั่นคงให้ยืนข้างเรานะ', 'มุขเสี่ยวคิดไม่ทันคิดถึงเธอก่อนแล้วกันมันง่ายดี 😚😚😚', 'วันนี้เธอเป็นหวานตา แต่วันข้างหน้าเธอเป็นหวานใจ', 'การบ้านอ่ะ ส่งให้ครู แต่เลิฟยู ส่งให้เธอ 💖💖💖', 'เบื่อแล้วหน้าฝนแต่ที่สนอะหน้าเธอ', 'เด็กคืออนาคตของชาติ แต่ถ้าไม่ผิดพลาดเราคืออนาคตของเธอ', 'หน้าตาไม่ฟรุ้งฟริ้งเลยถูกทิ้งให้ฟุ้งซ่าน', 'ตอนนี้ยังโสด ส่วนกิจกรรมโปรด คือ อ่อย 😎']

@client.event
async def on_ready():
    print('Logged in as')
    print(client.user.name)
    print(client.user.id)
    print('------')

@client.event
async def on_message(message):
    # ignore own message
    msg = message.content
    print("sentence = " + message.content)
    if message.author == client.user:
        return
    elif msg.lower() in greeting :
        return await client.send_message(message.channel, 'ดี {0.author.mention}'.format(message))
    elif msg.lower() in joke:
        return await client.send_message(message.channel, "{} {}".format(message.author.mention, choice(jokes)))
    else:
        outmsg = bot.talk(msg.lower())
        if outmsg =="noMatch":
            return await client.send_message(message.channel, msg + "หรออ 55555" )
        else:
            return await client.send_message(message.channel, outmsg)
        

client.run(token)
