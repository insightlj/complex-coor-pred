def send_email(content):
    import smtplib
    from email.mime.text import MIMEText
    mail_host = 'smtp.163.com'
    mail_user = 'nasainsight'
    mail_pass = 'TGKBWWXIJFQZWCGX'
    sender = 'nasainsight@163.com'
    receivers = ['nasainsight@163.com']
    message = MIMEText(content,'plain','utf-8')
    message['Subject'] = 'main.py运行结束，请查看'
    message['From'] = sender
    message['To'] = receivers[0]

    try:
        smtpObj = smtplib.SMTP()
        smtpObj.connect(mail_host,25)
        smtpObj.login(mail_user,mail_pass)
        smtpObj.sendmail(
            sender,receivers,message.as_string())
        smtpObj.quit()
    except smtplib.SMTPException as e:
        print('error',e) 
