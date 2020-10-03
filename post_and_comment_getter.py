import praw


# first get an instance of reddit with which to do all requests n staf
# also this info DEF should NOT be hard coded
reddit = praw.Reddit(client_id="D8OOb1-ucumfsQ",client_secret="WTOFFjDLkOybenai5vqXHffxnqs",user_agent="helloWorldAgent")

def get_post_by_id(idstr):
    return reddit.submission(id=idstr)

def get_permalink_from_commentid(idstr):
    return reddit.comment(id=idstr).permalink()

#helpful info:
#format of opiates comment data is
#indexes = {0:"body",1:"score_hidden",2:"archived",3:"name",4:"author",5:"author_flair_text",6:"downs",7:"created_utc",8:"subreddit_id",9:"link_id",10:"parent_id",11:"score",12:"retrieved_on",13:"controversiality",
#        14:"gilded",15:"id",16:"subreddit",17:"ups", 18:"distinguished",19:"author_flair_css_class",20:"removal_reason"}
# if the link_id and parent_id field are the same, its a top level comment
# posts have 't3_' inserted at the beginning of their id field, and comments have 't1_'
