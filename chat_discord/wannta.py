import discord
from chatbot import *

sess,enc_vocab,inv_dec_vocab,model = initchat()

client = discord.Client()
# bot = talk.talkWithMe()
#Token
token = "NDQwMDgwMjY5NTMwMjM0ODgw.DccgWw.8htsOsNsOm-Iwv-PWshAeQR7ezc"

#Type of Greeting
greeting = ['hello','hi','สวัสดี','ดี']
name = 'ชื่อ'

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
    print(msg)
    if message.author == client.user:
        return
    elif msg.lower() in greeting :
        return await client.send_message(message.channel, 'ดี {0.author.mention}'.format(message))
    elif name in msg.lower() :
        return await client.send_message(message.channel, 'สวัสดีเราชื่อหวานตานะ')
    else:
        outmsg = chat(sess, msg,enc_vocab,inv_dec_vocab,model)
#         if outmsg =="noMatch":
#             return await client.send_message(message.channel, msg + "หรออ 55555" )
#         else:
        return await client.send_message(message.channel, outmsg)


client.run(token)
