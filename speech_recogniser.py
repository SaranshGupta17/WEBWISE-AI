import speech_recognition as sr

r = sr.Recognizer()



def mic():
    with sr.Microphone()  as source:

        print("speak anything :")

        audio = r.listen(source)



        try:

            text = r.recognize_google(audio)
            
            query = '{}'.format(text)
            return query
        except:

            return "sorry can not recognize your voice"